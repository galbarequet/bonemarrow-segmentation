
import contextlib
import os
import pathlib
import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg
import numpy as np
from PIL import Image

import config
import model_runner
import network_utils
import utils


_lock = RendererAgg.lock

# General Constants
WEIGHTS_DIR = r'weights'
MODEL_WEIGHTS = os.path.join(WEIGHTS_DIR, 'bms_model.pt')


@st.cache(allow_output_mutation=True)
def load_model():
    model_base_dir = pathlib.Path(WEIGHTS_DIR)
    model_base_dir.mkdir(exist_ok=True)

    weights_path = pathlib.Path(MODEL_WEIGHTS)
    if not weights_path.exists():
        with st.spinner("Downloading model weights... this may take a few minutes."
                        " (~260 MB) Please don't interrupt it."):
            network_utils.download_file_from_google_drive(config.MODEL_WEIGHTS_DRIVE_ID, MODEL_WEIGHTS)

    model = model_runner.ModelRunner(
        weights_path=MODEL_WEIGHTS,
        crop_size=config.DEFAULT_CROP_SIZE,
        step_size=config.DEFAULT_STEP_SIZE
    )
    return model


@contextlib.contextmanager
def use_progress(model, progress_handler):
    model.progress_event.on_change += progress_handler
    try:
        yield
    finally:
        model.progress_event.on_change -= progress_handler


def segment_image():
    columns = st.columns(2)
    with columns[1]:
        st.markdown('<br>', unsafe_allow_html=True)
        st.info('**PRIVACY POLICY**: uploaded images are never saved or stored. They are held entirely '
                'within memory for prediction and discarded after the final results are displayed. ')

    f = columns[0].file_uploader("Please Select to Upload an Image", type=['png'])
    if f is None:
        return
    with f:
        image = np.array(Image.open(f))

    finished_processing = False
    bar = st.progress(0)
    progress_text = st.text('Please wait for magic to happen! This may take up to a minute.')

    def _progress_handler(percentage):
        if not finished_processing:
            display_progress = f'Processed {percentage}% of the image.'
            if percentage == 100:
                display_progress += ' Wait for segmented image to be displayed.'
        else:
            display_progress = f'Image processing complete!'

        progress_text.text(display_progress)
        bar.progress(percentage)

    columns = st.columns(3)

    columns[0].image(image, caption="Selected Input", output_format='PNG')

    model = load_model()
    with use_progress(model, _progress_handler):
        prediction = model.predict(image)
        finished_processing = True
        segmented_image = utils.create_seg_image(prediction)
        _progress_handler(100)

    columns[1].image(segmented_image, caption="Predicted Segmentation", output_format='PNG')
    st.markdown(
        network_utils.get_image_download_link(segmented_image, 'output.png', 'Download segmented image!'),
        unsafe_allow_html=True
    )

    # Note: display density of different tissues in the segmented pixels
    with _lock:
        fig, ax = utils.create_density_figure(prediction)
        with columns[2]:
            st.subheader('Bone Marrow Density From Total Mass:')
            st.pyplot(fig)


def main():
    st.set_page_config(page_title='Bone Marrow App', layout='wide')
    st.title('Bone Marrow Segmentation Application')
    st.write('Bone Marrow Segmentation (BMS) is the problem domain of automatic '
             'segmentation of different tissues in bone-marrow biopsy samples.')
    st.write('BMS is relevant for medical research purposes. In particular, BMS is used for researching'
             ' the corelation between the bone density in the biopsy and certain diseases.')
    st.write('**Our goal is to automate segmentation of different tissues in a bone-marrow biopsy sample.**')

    segment_image()


if __name__ == "__main__":
    main()
