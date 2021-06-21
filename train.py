import argparse
import os
import pickle

import numpy as np
import torch
from torch.nn import BCELoss
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from hannahmontananet import HannahMontanaNet

from dataset import BoneMarrowDataset as Dataset
from transform import transforms
from utils import dsc, pred_image_crop


def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    hannahmontana_net = HannahMontanaNet()
    hannahmontana_net.to(device)

    loss_func = BCELoss()
    best_validation_loss = 1.0

    optimizer = optim.Adam(hannahmontana_net.parameters(), lr=args.lr)

    loss_train = []
    loss_valid = []
    loss_train_mean = []
    loss_valid_mean = []
    validation_fat_layer_dsc = []
    validation_bone_layer_dsc = []

    step = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        if epoch % 10 == 0:
            save_stats(args, validation_bone_layer_dsc, validation_fat_layer_dsc, loss_train_mean, loss_valid_mean)

        for phase in ["train", "valid"]:
            if phase == "train":
                hannahmontana_net.train()
            else:
                hannahmontana_net.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if phase == "train":
                        y_pred = hannahmontana_net(x)
                    else:
                        y_pred = pred_image_crop(x, hannahmontana_net, args.crop_size, device)

                    loss = loss_func(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

            if phase == "train":
                loss_train_mean.append(np.mean(loss_train))
                loss_train = []

            if phase == "valid":
                bone_dsc, fat_dsc = zip(*calculate_dsc(
                        validation_pred,
                        validation_true)
                )
                validation_bone_layer_dsc.append(np.mean(bone_dsc))
                validation_fat_layer_dsc.append(np.mean(fat_dsc))
                print('validation loss is {}'.format(loss.item()))
                print('mean bone dsc {}'.format(np.mean(bone_dsc)))
                print('mean fat dsc {}'.format(np.mean(fat_dsc)))
                if loss.item() < best_validation_loss:
                    best_validation_loss = loss.item()
                    torch.save(hannahmontana_net.state_dict(), os.path.join(args.weights, "unet.pt"))
                loss_valid_mean.append(np.mean(loss_valid))
                loss_valid = []
                torch.save(hannahmontana_net.state_dict(), os.path.join(args.weights, "latest_unet.pt"))

    print("Best validation loss: {:4f}".format(best_validation_loss))
    save_stats(args, validation_bone_layer_dsc, validation_fat_layer_dsc, loss_train_mean, loss_valid_mean)


def save_stats(args, bone_layer_dsc, fat_layer_dsc, train_loss, valid_loss):
    stats = {}
    stats['bone_dsc'] = bone_layer_dsc
    stats['fat_dsc'] = fat_layer_dsc
    stats['train_loss'] = train_loss
    stats['valid_loss'] = valid_loss
    with open(os.path.join(args.stats, 'stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)



def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args):
    #TODO: change back to the original code
    train = Dataset(
        images_dir=args.images,
        subset="train",
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5, crop=args.crop_size, color_applay=args.color_apply),
    )
    valid = Dataset(
        images_dir=args.images,
        subset="validation",
        random_sampling=False,
    )
    return train, valid


def calculate_dsc(validation_pred, validation_true):
    dsc_list = []
    for i in range(len(validation_pred)):
        y_pred = validation_pred[i]
        y_true = validation_true[i]
        dsc_list.append(dsc(y_pred, y_true))
    return dsc_list


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.stats, exist_ok=True)


if __name__ == "__main__":
    # TODO: change back to the original code
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of bone marrow"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="number of epochs to train (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00001,
        help="initial learning rate (default: 0.00001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--stats", type=str, default="./stats", help="folder to save stats"
    )
    parser.add_argument(
        "--images", type=str, default="./data_samples", help="root folder with images"
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=1024,
        help="crop size",
    )
    parser.add_argument(
        "--color-apply",
        type=float,
        default=1,
        help="colors",
    )
    args = parser.parse_args()
    main(args)
