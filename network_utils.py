import base64
import io
import imageio
import requests


def _save_response_content(response, destination):
    chunk_size = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


# based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def download_file_from_google_drive(file_id, destination):
    google_url = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    params = {'id': file_id, 'confirm': 't'}
    response = session.get(google_url, params=params, stream=True)

    _save_response_content(response, destination)


# Note: based on https://discuss.streamlit.io/t/how-to-download-image/3358/2
def get_image_download_link(image, filename, text):
    buffered = io.BytesIO()
    imageio.imwrite(buffered, image, format='png')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'

