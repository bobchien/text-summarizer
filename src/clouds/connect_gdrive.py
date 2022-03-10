### Reference
# StackOverflow: https://stackoverflow.com/a/39225039
# Streamlit discussion: https://discuss.streamlit.io/t/how-to-download-large-model-files-to-the-sharing-app/7160

import requests
from pathlib import Path

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id':file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id':file_id, 'confirm':token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)    

def gdown_file_from_google_drive(file_id, file_path):
    import gdown
    
    url = 'https://drive.google.com/uc?id='+file_id
    gdown.download(url, file_path, quiet=False)
    