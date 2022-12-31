import argparse
import contextlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imsave
from tqdm import tqdm

import config
import model_runner
import network_utils
import utils


# General Constants
_WEIGHTS_DIR = Path(__file__).parent.joinpath('weights')
_MODEL_WEIGHTS = _WEIGHTS_DIR.joinpath('bms_model.pt')


def _load_model(weights_path, crop_size, step_size):
    if not weights_path.exists():
        weights_path.parent.mkdir(exist_ok=True)
        print("Downloading model weights... this may take a few minutes.")
        print("(~260 MB) Please don't interrupt it.")
        network_utils.download_file_from_google_drive(config.MODEL_WEIGHTS_DRIVE_ID, weights_path)

    model = model_runner.ModelRunner(weights_path, crop_size, step_size)
    return model


class ProgressHandler:
    def __init__(self, callback):
        self._callback = callback
        self._last_percentage = 0

    def __call__(self, percentage):
        self._callback(percentage - self._last_percentage)
        self._last_percentage = percentage


@contextlib.contextmanager
def use_progress(model):
    with tqdm(total=100) as progress_bar:
        progress_handler = ProgressHandler(progress_bar.update)
        model.progress_event.on_change += progress_handler
        try:
            yield
        finally:
            model.progress_event.on_change -= progress_handler


def segment_image(model, image_path, output_directory):
    with Image.open(image_path) as f:
        image = np.array(f)

    with use_progress(model):
        prediction = model.predict(image)
        segmented_image = utils.create_seg_image(prediction)

    segmented_image_path = output_directory.joinpath(image_path.with_name(f'{image_path.stem}_prediction.png').name)
    stats_image_path = output_directory.joinpath(image_path.with_name(f'{image_path.stem}_stats.png').name)

    imsave(segmented_image_path.as_posix(), segmented_image)

    # Note: display density of different tissues in the segmented pixels
    fig, ax, densities = utils.create_density_figure(prediction)

    print(f'Bone Marrow Density From Total Mass for image {image_path.name}')
    for density_type, density in densities.items():
        print(f'\t{density_type}: {density:.3f}%')
    print()

    fig.suptitle('Bone Marrow Density From Total Mass')
    fig.savefig(stats_image_path)
    plt.close(fig)


def main(args):
    model = _load_model(Path(args.weights), args.crop_size, args.step_size)

    Path(args.output).mkdir(exist_ok=True)

    for image_path in Path(args.images).glob('*.png'):
        segment_image(model, image_path, Path(args.output))


def _create_args():
    parser = argparse.ArgumentParser(
        description="Inference for segmentation of bone marrow"
    )
    parser.add_argument(
        "--weights",
        help="path to weights file",
        default=str(_MODEL_WEIGHTS),
    )
    parser.add_argument(
        "--images",
        required=True,
        help="bone-marrow biopsy samples directory"
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=config.DEFAULT_CROP_SIZE,
        help=f"target crop size for the sliding window when segmenting (default: {config.DEFAULT_CROP_SIZE})",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=config.DEFAULT_STEP_SIZE,
        help=f"target step size for the sliding window (default: {config.DEFAULT_STEP_SIZE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="folder for saving images' predictions",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(_create_args())
