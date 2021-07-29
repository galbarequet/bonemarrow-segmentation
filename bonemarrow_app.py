import model_runner
import cv2
import numpy as np
import os
import pandas as pd
import pathlib
from PIL import Image
import streamlit as st
import time
import urllib.request

# General Constants
# CR: (GB) change this
BASE_DIR = r'D:\Gali\university\tau\ML workshop\weights'
# CR: (GB) change this
MODEL_WEIGHTS = os.path.join(BASE_DIR, 'latest_unet_1000_00001_nc.pt')
DEFAULT_CROP_SIZE = 256
DEFAULT_STEP_SIZE = 128

IMAGE_DISPLAY_SIZE = (330, 330)


# Constants for sidebar dropdown
SIDEBAR_OPTION_PROJECT_INFO = "Show Project Info"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload an Image"
SIDEBAR_OPTION_MEET_TEAM = "Meet the Team"

SIDEBAR_OPTIONS = [SIDEBAR_OPTION_PROJECT_INFO, SIDEBAR_OPTION_UPLOAD_IMAGE, SIDEBAR_OPTION_MEET_TEAM]


@st.cache(allow_output_mutation=True)
def load_model():
    model_base_dir = pathlib.Path(BASE_DIR)
    model_base_dir.mkdir(exist_ok=True)

    weights_path = pathlib.Path(MODEL_WEIGHTS)
    if not weights_path.exists():
        with st.spinner("Downloading model weights... this may take a few minutes. (~260 MB) Please don't interrupt it."):
            # CR: (GB) fix this
            # download_file(url=MODEL_WEIGHTS_DEPLOYMENT_URL, local_filename=MODEL_WEIGHTS)
            pass

    model = model_runner.ModelRunner(weights_path=MODEL_WEIGHTS, crop_size=DEFAULT_CROP_SIZE,
                                     step_size=DEFAULT_STEP_SIZE)
    return model


def get_file_content_as_string(path):
    # CR: (GB) change this
    url = 'https://google.com'
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def show_project_info():
    st.sidebar.success("Project information showing on the right!")
    st.write(get_file_content_as_string("Project_Info.md"))


def segment_image():
    st.sidebar.info('PRIVACY POLICY: uploaded images are never saved or stored. They are held entirely within memory for prediction \
                    and discarded after the final results are displayed. ')

    f = st.sidebar.file_uploader("Please Select to Upload an Image", type=['png', 'jpg', 'jpeg', 'tiff', 'gif'])
    if f is None:
        return

    with f:
        st.sidebar.write('Please wait for the magic to happen! This may take up to a minute.')
        st.beta_set_page_config(layout="wide")
        left_column, right_column = st.beta_columns(2)

        image = Image.open(f)

        model = load_model()
        left_column.image(np.array(image), caption="Selected Input")

        segmented_image = model.run_segmentation(np.array(image))
        right_column.image(segmented_image, caption="Predicted Segmentation")
        'done'


def show_team_info():
    st.subheader("Our Team")

    'Gal Barequet'
    'Itay Kalev'
    'Sagi Tauber'

    st.sidebar.success('Hope you had a great time :)')


def main():
    mode_to_handler = {
        SIDEBAR_OPTION_PROJECT_INFO: show_project_info,
        SIDEBAR_OPTION_UPLOAD_IMAGE: segment_image,
        SIDEBAR_OPTION_MEET_TEAM: show_team_info
    }

    st.title('Bone Marrow App')
    st.sidebar.write(" ------ ")

    st.sidebar.title("Explore the Following")
    app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)

    handler = mode_to_handler.get(app_mode)
    if handler is not None:
        handler()
    else:
        raise ValueError('Selected sidebar option is not implemented. Please open an issue on Github: https://github.com/galbarequet/bonemarrow-segmentation')


if __name__ == "__main__":
    main()
