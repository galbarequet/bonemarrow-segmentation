import base64
import io
import imageio
import model_runner
import numpy as np
import os
import pathlib
from PIL import Image
import streamlit as st

# General Constants
# CR: (GB) change this
BASE_DIR = r'D:\Gali\university\tau\ML workshop\weights'
# CR: (GB) change this
MODEL_WEIGHTS = os.path.join(BASE_DIR, 'latest_unet_1000_00001_nc.pt')
DEFAULT_CROP_SIZE = 256
DEFAULT_STEP_SIZE = 128

# CR: (GB) remove this
IMAGE_DISPLAY_SIZE = (330, 330)


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


# Note: based on https://discuss.streamlit.io/t/how-to-download-image/3358/2
def get_image_download_link(image, filename, text):
    buffered = io.BytesIO()
    imageio.imwrite(buffered, image, format='png')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'


def segment_image():
    columns = st.beta_columns(2)
    with columns[1]:
        st.markdown('<br>', unsafe_allow_html=True)
        st.info('**PRIVACY POLICY**: uploaded images are never saved or stored. They are held entirely '
                'within memory for prediction and discarded after the final results are displayed. ')

    f = columns[0].file_uploader("Please Select to Upload an Image", type=['png'])
    if f is None:
        return
    with f:
        image = np.array(Image.open(f))

    bar = st.progress(0)
    progress_text = st.text('Please wait for magic to happen! This may take up to a minute.')

    def _progress_handler(percentage):
        display_progress = f'Finished processing {percentage}% of the image.'
        if percentage == 100:
            display_progress += ' Wait for segmented image to be displayed.'

        progress_text.text(display_progress)
        bar.progress(percentage)

    left_column, right_column = st.beta_columns(2)

    left_column.image(image, caption="Selected Input", output_format='PNG')

    model = load_model()
    model.progress_event.on_change += _progress_handler
    segmented_image = model.run_segmentation(image)
    model.progress_event.on_change -= _progress_handler

    right_column.image(segmented_image, caption="Predicted Segmentation", output_format='PNG')
    st.markdown(get_image_download_link(segmented_image, 'output.png', 'Download segmented image!'),
                unsafe_allow_html=True)
    'done'


def main():
    st.set_page_config(page_title='Bone Marrow App', layout='wide')
    st.title('Bone Marrow Segmentation Application')
    st.write('Bone Marrow Segmentation (BMS) is the problem domain of automatic '
             'segmentation of different tissues in bone-marrow biopsy samples.')
    st.write('BMS is relevant for medical research purposes. In particular, BMS is used for researching'
             ' the co-relation between the relative volume of fat/bone tissue in the biopsy and diseases.')
    st.write('**Our goal is to automate segmentation of different tissues in a bone-marrow biopsy sample.**')

    segment_image()


if __name__ == "__main__":
    main()
