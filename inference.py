import argparse
import json
import os
from PIL import Image
import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BoneMarrowDataset as Dataset
from utils import create_seg_image, dsc, outline, create_error_image, calculate_bonemarrow_density_error

from hannahmontananet import HannahMontanaNet
import sliding_window
matplotlib.use('Agg')


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

        sliding_window_predictor = sliding_window.SlidingWindow(net, args.crop_size, args.step_size)

        input_list = []
        pred_list = []
        true_list = []

        for i, data in tqdm(enumerate(loader)):
            print(i)
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            # y_pred = net(x)
            y_pred = sliding_window_predictor.predict_image(x)
            y_pred_np = y_pred.detach().cpu().numpy()
            pred_list.extend([np.argmax(y_pred_np[s], axis=0) for s in range(y_pred_np.shape[0])])

            y_true_np = y_true.detach().cpu().numpy()
            true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

            x_np = x.detach().cpu().numpy()
            input_list.extend([x_np[s] for s in range(x_np.shape[0])])

    n = len(input_list)

    dsc_background_dist, dsc_bone_dist, dsc_fat_dist, dsc_tissue_dist = dsc_distribution(pred_list, true_list)
    dsc_density_error = calc_bone_density_error_distribution(pred_list, true_list)

    dsc_background_dist_plot = plot_dsc(dsc_background_dist)
    imsave(os.path.join(args.figure, 'dsc_backgorund.png'), dsc_background_dist_plot)
    dsc_bone_dist_plot = plot_dsc(dsc_bone_dist)
    imsave(os.path.join(args.figure, 'dsc_bone.png'), dsc_bone_dist_plot)
    dsc_fat_dist_plot = plot_dsc(dsc_fat_dist)
    imsave(os.path.join(args.figure, 'dsc_fat.png'), dsc_fat_dist_plot)
    dsc_tissue_dist_plot = plot_dsc(dsc_tissue_dist)
    imsave(os.path.join(args.figure, 'dsc_tissue.png'), dsc_tissue_dist_plot)
    imsave(os.path.join(args.figure, 'dsc_tissue.png'), dsc_tissue_dist_plot)
    density_error_dist_plot = plot_dsc(dsc_density_error)
    imsave(os.path.join(args.figure, 'density_error.png'), density_error_dist_plot)

    for p in range(n):
        x = input_list[p].transpose(1, 2, 0).astype(np.uint8)
        y_pred = pred_list[p]
        y_true = true_list[p]
        image_shape = y_pred[0].shape

        # Note: calculates the confusion matrix like Moni described it.
        current_true_list = [y_true[0], y_true[1], np.ones(image_shape) - y_true[0] - y_true[1]]
        current_pred_list = [y_pred[0], y_pred[1], np.ones(image_shape) - y_pred[0] - y_pred[1]]
        confusion_matrix = [[0] * len(current_pred_list) for i in range(len(current_true_list))]
        for i in range(len(current_true_list)):
            for j in range(len(current_pred_list)):
                confusion_matrix[i][j] = int(np.sum(current_true_list[i] * current_pred_list[j]))
        
        original_filename = loader.dataset.names[p].rsplit('.')[0]
        folder_path = os.path.join(args.predictions, original_filename)
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, f'stats - {original_filename}.json'), 'w', encoding='utf-8') as f:
            json.dump(confusion_matrix, f, ensure_ascii=False, indent=4)

        # Note: calculate the relative volume of bone/fat from true background size
        # percentages = {
        #     'true': {},
        #     'pred': {},
        #     'diff': {}
        # }
        # true_background_size = np.count_nonzero(y_true[0] + y_true[1])
        # for i, label_type in enumerate(['bone', 'fat']):
        #     percentages['pred'][label_type] = 100 * np.count_nonzero(y_pred[i]) / true_background_size
        #     percentages['true'][label_type] = 100 * np.count_nonzero(y_true[i]) / true_background_size
        #     percentages['diff'][label_type] = percentages['pred'][label_type] - percentages['true'][label_type]

        # original_filename = loader.dataset.names[p].rsplit('.')[0]
        # folder_path = os.path.join(args.predictions, original_filename)
        # os.makedirs(folder_path, exist_ok=True)
        # with open(os.path.join(folder_path, f'stats - {original_filename}.json'), 'w', encoding='utf-8') as f:
        #     json.dump(percentages, f, ensure_ascii=False, indent=4)

        # Sagi's way
        folder_path = os.path.join(args.predictions, original_filename)
        os.makedirs(folder_path, exist_ok=True)

        imsave(os.path.join(folder_path, "raw.png"), x)
        imsave(os.path.join(folder_path, "pred.png"), create_seg_image(y_pred))
        imsave(os.path.join(folder_path, "true.png"), create_seg_image(y_true))

        imsave(os.path.join(folder_path, "background_error.png"), create_error_image(y_pred, y_true, 0))
        imsave(os.path.join(folder_path, "bones_error.png"), create_error_image(y_pred, y_true, 1))
        imsave(os.path.join(folder_path, "fat_error.png"), create_error_image(y_pred, y_true, 2))
        imsave(os.path.join(folder_path, "tissue_error.png"), create_error_image(y_pred, y_true, 3))


def data_loader(args):
    dataset = Dataset(
        images_dir=args.images,
        subset="validation",
        random_sampling=False,
        validation_cases=7,
        fat_overrides_bone=args.fat_overrides
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=1,
    )
    return loader


def dsc_distribution(pred_list, true_list):
    n = len(pred_list)
    dsc_background_dict = {}
    dsc_bone_dict = {}
    dsc_fat_dict = {}
    dsc_tissue_dict = {}
    for p in range(n):
        y_pred = pred_list[p]
        y_true = true_list[p]
        dsc_background_dict[p], dsc_bone_dict[p], dsc_fat_dict[p], dsc_tissue_dict[p] = \
            dsc(y_pred, y_true)
    return dsc_background_dict, dsc_bone_dict, dsc_fat_dict, dsc_tissue_dict

def calc_bone_density_error_distribution(pred_list, true_list):
    n = len(pred_list)
    bone_density_error_dict = {}
    for p in range(n):
        y_pred = pred_list[p]
        y_true = true_list[p]
        bone_density_error_dict[p] = calculate_bonemarrow_density_error(y_pred, y_true)
    return bone_density_error_dict


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
        help="folder for saving images with prediction outlines",
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

    main(parser.parse_args())
