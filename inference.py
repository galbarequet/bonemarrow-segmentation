import argparse
import os
from PIL import Image

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BoneMarrowDataset as Dataset
from utils import create_seg_image, dsc, outline, pred_image_crop

from hannahmontananet import HannahMontanaNet


def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader = data_loader(args)

    with torch.set_grad_enabled(False):
        net = HannahMontanaNet(out_channels=Dataset.out_channels)
        state_dict = torch.load(args.weights, map_location=device)
        net.load_state_dict(state_dict)
        net.eval()
        net.to(device)

        input_list = []
        pred_list = []
        true_list = []

        for i, data in tqdm(enumerate(loader)):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            # y_pred = net(x)
            y_pred = pred_image_crop(x, net, args.crop_size, step_size=args.step_size)
            y_pred_np = y_pred.detach().cpu().numpy()
            y_pred_np = np.round(y_pred_np).astype(np.int)
            pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

            y_true_np = y_true.detach().cpu().numpy()
            true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

            x_np = x.detach().cpu().numpy()
            input_list.extend([x_np[s] for s in range(x_np.shape[0])])

    n = len(input_list)

    dsc_bone_dist, dsc_fat_dist = dsc_distribution(pred_list, true_list)

    dsc_bone_dist_plot = plot_dsc(dsc_bone_dist)
    imsave(os.path.join(args.figure, 'dsc_bone.png'), dsc_bone_dist_plot)
    dsc_fat_dist_plot = plot_dsc(dsc_fat_dist)
    imsave(os.path.join(args.figure, 'dsc_fat.png'), dsc_fat_dist_plot)

    for p in range(n):
        x = input_list[p].transpose(1,2,0).astype(np.uint8)
        y_pred = pred_list[p]
        y_true = true_list[p]

        original_filename = loader.dataset.names[p].rsplit('.')[0]
        # Sagi's way
        folder_path = os.path.join(args.predictions, original_filename)
        os.makedirs(folder_path, exist_ok=True)
        imsave(os.path.join(folder_path, "raw.png"), x)
        imsave(os.path.join(folder_path, "pred.png"), create_seg_image(y_pred))
        imsave(os.path.join(folder_path, "true.png"), create_seg_image(y_true))

        # Outline way
        folder_path = os.path.join(args.predictions, "outline")
        image = x
        image = outline(image, y_pred[0], color=[255, 0, 0])
        image = outline(image, y_true[0], color=[0, 255, 0])
        filename = "{}.png".format(p)
        filepath = os.path.join(folder_path, filename)
        imsave(filepath, image)
        print("finished: {}".format(p))


def data_loader(args):
    dataset = Dataset(
        images_dir=args.images,
        subset="validation",
        random_sampling=False,
        validation_cases=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=1
    )
    return loader


def dsc_distribution(pred_list, true_list):
    n = len(pred_list)
    dsc_bone_dict = {}
    dsc_fat_dict = {}
    for p in range(n):
        y_pred = pred_list[p]
        y_true = true_list[p]
        dsc_bone_dict[p], dsc_fat_dict[p] = dsc(y_pred, y_true)
    return dsc_bone_dict, dsc_fat_dict


def plot_dsc(dsc_dist):
    y_positions = np.arange(len(dsc_dist))
    dsc_dist = sorted(dsc_dist.items(), key=lambda x: x[1])
    values = [x[1] for x in dsc_dist]
    labels = [x[0] for x in dsc_dist]
    fig = plt.figure(figsize=(12, 8))
    canvas = FigureCanvasAgg(fig)
    plt.barh(y_positions, values, align="center", color="skyblue")
    plt.yticks(y_positions, labels)
    plt.xticks(np.arange(0.0, 1.0, 0.1))
    plt.xlim([0.0, 1.0])
    plt.gca().axvline(np.mean(values), color="tomato", linewidth=2)
    plt.gca().axvline(np.median(values), color="forestgreen", linewidth=2)
    plt.xlabel("Dice coefficient", fontsize="x-large")
    plt.gca().xaxis.grid(color="silver", alpha=0.5, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))


def makedirs(args):
    os.makedirs(args.predictions, exist_ok=True)
    os.makedirs(args.figure, exist_ok=True)
    os.makedirs(os.path.join(args.predictions, "outline"), exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference for segmentation of bone marrow"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="path to weights file"
    )
    parser.add_argument(
        "--images", type=str, default="./data_samples", help="root folder with images"
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="target crop size for the sliding window when segmenting (default: 256)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=128,
        help="target step size for the sliding window (default: 128)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="./predictions",
        help="folder for saving images with prediction outlines",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="./dsc",
        help="filename for DSC distribution folder",
    )

    args = parser.parse_args()
    main(args)
