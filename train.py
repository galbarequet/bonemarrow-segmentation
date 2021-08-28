import argparse
import os
import pickle

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from hannahmontananet import HannahMontanaNet

from dataset import BoneMarrowDataset as Dataset
from transform import transforms
from utils import dsc, calculate_bonemarrow_density_error
import sliding_window


def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid, loader_test = data_loaders(args)
    loaders = {'train': loader_train, 'valid': loader_valid}
    phase_samples = {
        'train': loader_train.dataset.names,
        'valid': loader_valid.dataset.names,
        'test': loader_test.dataset.names
    }
    print("validation set: {}".format(loader_valid.dataset.names))
    print("test set: {}".format(loader_test.dataset.names))

    hannahmontana_net = HannahMontanaNet()
    hannahmontana_net.to(device)
    sliding_window_predictor = sliding_window.SlidingWindow(hannahmontana_net, args.crop_size, args.step_size)

    loss_func = CrossEntropyLoss()
    best_validation_loss = 1.0
    best_bone_dsc = 0
    best_bone_density = 5000

    optimizer = optim.Adam(hannahmontana_net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.scheduler_factor,
                                                     patience=args.scheduler_patience)

    loss_train = []
    loss_valid = []
    loss_train_mean = []
    loss_valid_mean = []

    validation_density_error = []
    validation_background_dsc = []
    validation_bone_dsc = []
    validation_fat_dsc = []
    validation_tissue_dsc = []

    step = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        print(f'Epoch - {epoch}')

        save_stats(args, phase_samples, validation_density_error, loss_train_mean, loss_valid_mean,
                   validation_background_dsc, validation_bone_dsc, validation_fat_dsc, validation_tissue_dsc)

        for phase in ["train", "valid"]:
            if phase == "train":
                hannahmontana_net.train()
            else:
                if epoch % args.validate_every != 0:
                    break
                hannahmontana_net.eval()

            validation_pred = []
            validation_true = []

            for _, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if phase == "train":
                        y_pred = hannahmontana_net(x)
                    else:
                        y_pred = sliding_window_predictor.predict_image(x)

                    loss = loss_func(y_pred, y_true)

                    if phase == "valid":
                        scheduler.step(loss)
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()

                        validation_pred.extend(
                            [np.argmax(y_pred_np[s], axis=0) for s in range(y_pred_np.shape[0])]
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
                bone_density_error = calculate_bone_density_error_validation(
                        validation_pred,
                        validation_true)

                background_dsc, bone_dsc, fat_dsc, tissue_dsc = zip(*calculate_dsc(validation_pred, validation_true))
                mean_bone_density_error = np.mean(bone_density_error)
                mean_background_dsc = np.mean(background_dsc)
                mean_bone_dsc = np.mean(bone_dsc)
                mean_fat_dsc = np.mean(fat_dsc)
                mean_tissue_dsc = np.mean(tissue_dsc)

                validation_density_error.append(mean_bone_density_error)
                validation_background_dsc.append(mean_background_dsc)
                validation_bone_dsc.append(mean_bone_dsc)
                validation_fat_dsc.append(mean_fat_dsc)
                validation_tissue_dsc.append(mean_tissue_dsc)

                print('validation loss is {}'.format(loss.item()))
                print('mean bone density error is {}%'.format(mean_bone_density_error * 100))
                print('mean background dsc {}'.format(mean_background_dsc))
                print('mean bone dsc {}'.format(mean_bone_dsc))
                print('mean fat dsc {}'.format(mean_fat_dsc))
                print('mean tissue dsc {}'.format(mean_tissue_dsc))

                if loss.item() < best_validation_loss:
                    best_validation_loss = loss.item()
                    torch.save(hannahmontana_net.state_dict(), os.path.join(args.weights, "net_best_loss.pt"))
                if mean_bone_density_error < best_bone_density:
                    best_bone_density = mean_bone_density_error
                    torch.save(hannahmontana_net.state_dict(), os.path.join(args.weights, "net_best_density.pt"))
                if mean_bone_dsc > best_bone_dsc:
                    best_bone_dsc = mean_bone_dsc
                    torch.save(hannahmontana_net.state_dict(), os.path.join(args.weights, "net_best_bone_dsc.pt"))

                loss_valid_mean.append(np.mean(loss_valid))
                loss_valid = []
                try_save_model(args, hannahmontana_net.state_dict(), os.path.join(args.weights, "latest_net.pt"))

    print("Best validation loss: {:4f}".format(best_validation_loss))
    save_stats(args, phase_samples, validation_density_error, loss_train_mean, loss_valid_mean,
               validation_background_dsc,
               validation_bone_dsc, validation_fat_dsc, validation_tissue_dsc)


def try_save_model(args, obj, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(obj, path)

    except BaseException as e:
        print(f'Save failed. Exception: {e}')
        print('Saving to backup dir')
        try:
            os.makedirs(args.backup_dir, exist_ok=True)
            torch.save(obj, os.path.join(args.backup_dir, os.path.basename(path)))
        except BaseException as e:
            print(f'Save to backup dir failed. Exception: {e}')
            print('skipping save...')


def save_stats(args, dataset, validation_density_error, train_loss, valid_loss,
               background_dsc, bone_dsc, fat_dsc, tissue_dsc):
    stats = {
        'dataset': dataset,
        'bone_density_error': validation_density_error,
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'backgrounds_dsc': background_dsc,
        'bone_dsc': bone_dsc,
        'fat_dsc': fat_dsc,
        'tissue_dsc': tissue_dsc,
    }
    with open(os.path.join(args.stats, 'stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)


def data_loaders(args):
    dataset_train, dataset_valid, dataset_test = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        drop_last=True
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid, loader_test


def datasets(args):
    #TODO: change back to the original code
    train = Dataset(
        images_dir=args.images,
        subset="train",
        transform=transforms(
            scale=args.aug_scale,
            angle=args.aug_angle,
            flip_prob=0.5,
            crop=args.crop_size,
            color_apply=args.color_apply,
            elastic_apply=args.elastic_apply,
        ),
        fat_overrides_bone=args.fat_overrides,
    )
    valid = Dataset(
        images_dir=args.images,
        subset="validation",
        random_sampling=False,
        fat_overrides_bone=args.fat_overrides,
    )
    test = Dataset(
        images_dir=args.images,
        subset="test",
        random_sampling=False,
        fat_overrides_bone=args.fat_overrides,
    )
    return train, valid, test


def calculate_bone_density_error_validation(validation_pred, validation_true):
    error_list = []
    for i in range(len(validation_pred)):
        y_pred = validation_pred[i]
        y_true = validation_true[i]
        error_list.append(calculate_bonemarrow_density_error(y_pred, y_true))
    return error_list


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
    os.makedirs(args.backup_dir, exist_ok=True)


if __name__ == "__main__":
    # TODO: change back to the original code
    parser = argparse.ArgumentParser(
        description="Training model for segmentation of bone marrow"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="input batch size for training (default: 4)",
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
        default=0.0001,
        help="initial learning rate (default: 0.0001)",
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
        "--backup-dir", type=str, default="./backup", help="backup folder for saving model"
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
        default=180,
        help="rotation angle range in degrees for augmentation (default: 180)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="crop size",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=128,
        help="step size",
    )
    parser.add_argument(
        "--color-apply",
        type=float,
        default=0.25,
        help="colors",
    )
    parser.add_argument(
        "--elastic-apply",
        type=float,
        default=0.25,
        help="elastic transform probability",
    )
    parser.add_argument(
        "--fat-overrides",
        type=bool,
        default=True,
        help="does fat override bones?",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="The factor that the lr is reduced",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=10,
        help="Epochs without improvements before the scheduler reduces the lr",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=10,
        help="Validate every x train rounds",
    )
    main(parser.parse_args())
