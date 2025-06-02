import pandas as pd
import numpy as np
import os
import requests
import zipfile
from scripts.utils import root_dir

data_url = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
download_path = "data/ml-32m.zip"

def download_data(url, download_path):
    if not os.path.exists(download_path):
        print(f"Downloading data from {url}...")
        response = requests.get(url)
        with open(download_path, 'wb') as file:
            file.write(response.content)
        print("Download complete.")
    else:
        print("Data already downloaded.")

def load_data_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path))
    print(f"Data extracted to {os.path.dirname(zip_path)}")

    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"Removed zip file: {zip_path}")

download_path = os.path.join(root_dir, download_path)
download_data(data_url, download_path)
load_data_from_zip(download_path)



