import argparse
import json
import os
import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm
from baseline.baseline import segment_image

from bonemarrow_label import BoneMarrowLabel
from dataset import BoneMarrowDataset as Dataset
from utils import create_seg_image, dsc, create_error_image, calculate_bonemarrow_density_error

from hannahmontananet import HannahMontanaNet
import sliding_window


def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader = data_loader(args)
    names = [s.rsplit('.')[0] for s in loader.dataset.names]

    with torch.set_grad_enabled(False):
        if not args.baseline:
            net = HannahMontanaNet(out_channels=Dataset.out_channels)
            state_dict = torch.load(args.weights, map_location=device)
            net.load_state_dict(state_dict)
            net.eval()
            net.to(device)

            sliding_window_predictor = sliding_window.SlidingWindow(net, args.crop_size, args.step_size)

        dsc_background_dist = {}
        dsc_bone_dist = {}
        dsc_fat_dist = {}
        dsc_tissue_dist = {}
        density_error_dist = {}

        for i, data in tqdm(enumerate(loader)):
            x, y_true = data
            filename = names[i]
            print('segmenting file: {}'.format(filename))

            if args.baseline:
                x = x.detach().numpy()
                normal_image = x[0, ...].transpose(1, 2, 0).astype(np.uint8)
                y_pred_np = segment_image(normal_image)
                y_true_np = y_true.detach().numpy()
                x_np = x
            else:
                x, y_true = x.to(device), y_true.to(device)
                y_pred = sliding_window_predictor.predict_image(x)
                y_pred_np = y_pred.detach().cpu().numpy()[0, ...]
                y_pred_np = np.argmax(y_pred_np, axis=0)
                x_np = x.detach().cpu().numpy()
                y_true_np = y_true.detach().cpu().numpy()

            y_true_np = y_true_np[0, ...]
            x_np = x_np[0, ...].transpose(1, 2, 0).astype(np.uint8)
            save_image_stats(x_np, y_pred_np, y_true_np, args.predictions, filename)
            dsc_background_dist[filename], dsc_bone_dist[filename], dsc_fat_dist[filename], dsc_tissue_dist[filename] = \
                dsc(y_pred_np, y_true_np)
            density_error_dist[filename] = calculate_bonemarrow_density_error(y_pred_np, y_true_np)

    dsc_background_dist_plot = plot_param_dist(dsc_background_dist)
    imsave(os.path.join(args.figure, 'dsc_backgorund.png'), dsc_background_dist_plot)
    dsc_bone_dist_plot = plot_param_dist(dsc_bone_dist)
    imsave(os.path.join(args.figure, 'dsc_bone.png'), dsc_bone_dist_plot)
    dsc_fat_dist_plot = plot_param_dist(dsc_fat_dist)
    imsave(os.path.join(args.figure, 'dsc_fat.png'), dsc_fat_dist_plot)
    dsc_tissue_dist_plot = plot_param_dist(dsc_tissue_dist)
    imsave(os.path.join(args.figure, 'dsc_tissue.png'), dsc_tissue_dist_plot)
    density_error_dist_plot = plot_param_dist(density_error_dist, param_name='Density error')
    imsave(os.path.join(args.figure, 'density_error.png'), density_error_dist_plot)


def save_image_stats(x, y_pred, y_true, folder, name):
    """
        Saves the confusion matrix, segmentation image, and error images to a folder named name inside folder
    """
    folder_path = os.path.join(folder, name)
    os.makedirs(folder_path, exist_ok=True)

    # Saves the confusion matrix
    cm = calculate_confusion_matrix(y_pred, y_true)
    with open(os.path.join(folder_path, f'stats - {name}.json'), 'w', encoding='utf-8') as f:
        json.dump(cm, f, ensure_ascii=False, indent=4)

    # save segmented images and respective errors
    imsave(os.path.join(folder_path, "raw.png"), x)
    imsave(os.path.join(folder_path, "pred.png"), create_seg_image(y_pred))
    imsave(os.path.join(folder_path, "true.png"), create_seg_image(y_true))

    imsave(os.path.join(folder_path, "background_error.png"),
            create_error_image(y_pred, y_true, BoneMarrowLabel.BACKGROUND))
    imsave(os.path.join(folder_path, "bones_error.png"),
            create_error_image(y_pred, y_true, BoneMarrowLabel.BONE))
    imsave(os.path.join(folder_path, "fat_error.png"),
            create_error_image(y_pred, y_true, BoneMarrowLabel.FAT))
    imsave(os.path.join(folder_path, "tissue_error.png"),
            create_error_image(y_pred, y_true, BoneMarrowLabel.OTHER))


def calculate_confusion_matrix(y_pred, y_true):
    """
        Calculates the confusion matrix
    """
    true_list = [y_true == BoneMarrowLabel.BACKGROUND, y_true == BoneMarrowLabel.BONE,
                 y_true == BoneMarrowLabel.FAT, y_true == BoneMarrowLabel.OTHER]
    pred_list = [y_pred == BoneMarrowLabel.BACKGROUND, y_pred == BoneMarrowLabel.BONE,
                 y_pred == BoneMarrowLabel.FAT, y_pred == BoneMarrowLabel.OTHER]
    confusion_matrix = [[0] * len(pred_list) for i in range(len(true_list))]
    for i in range(len(true_list)):
        for j in range(len(pred_list)):
            confusion_matrix[i][j] = int(np.sum(true_list[i] * pred_list[j]))
    return confusion_matrix


def data_loader(args):
    dataset = Dataset(
        images_dir=args.images,
        subset="validation",
        random_sampling=False,
        validation_cases=None,
        fat_overrides_bone=args.fat_overrides
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=1,
    )
    return loader


def plot_param_dist(dist, param_name="Dice coefficient"):
    """
        Return the plot of the parameter distribution.
        Note: the parameter needs to get value from (0,1).
    """
    y_positions = np.arange(len(dist))
    dist = sorted(dist.items(), key=lambda x: x[1])
    values = [x[1] for x in dist]
    labels = [x[0] for x in dist]
    fig = plt.figure(figsize=(12, 8))
    canvas = FigureCanvasAgg(fig)
    plt.barh(y_positions, values, align="center", color="skyblue")
    plt.yticks(y_positions, labels)
    plt.xticks(np.arange(0.0, 1.0, 0.1))
    plt.xlim([0.0, 1.0])
    plt.gca().axvline(np.mean(values), color="tomato", linewidth=2)
    plt.gca().axvline(np.median(values), color="forestgreen", linewidth=2)
    plt.xlabel(param_name, fontsize="x-large")
    plt.gca().xaxis.grid(color="silver", alpha=0.5, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.frombuffer(s, np.uint8).reshape((height, width, 4))


def makedirs(args):
    os.makedirs(args.predictions, exist_ok=True)
    os.makedirs(args.figure, exist_ok=True)


if __name__ == "__main__":
    matplotlib.use('Agg')
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
        default=512,
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
        help="folder for saving images' predictions",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="./dsc",
        help="filename for DSC distribution folder",
    )
    parser.add_argument(
        "--fat-overrides",
        type=bool,
        default=True,
        help="does fat override bones?",
    )
    parser.add_argument(
        "--baseline",
        type=bool,
        default=False,
        help="run the baseline (True) or network (False)",
    )

    main(parser.parse_args())
