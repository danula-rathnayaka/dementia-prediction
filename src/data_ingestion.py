import os

import gdown

from src.utils import load_config


def ingest_data(config):
    """Downloads the dataset from Google Drive and saves it locally."""
    data_config = config['data']
    paths_config = config['paths']

    url = data_config['drive_url']
    output_path = paths_config['raw_data']

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Downloading data from: {url}")

    gdown.download(url, output_path, quiet=False)

    if os.path.exists(output_path):
        print(f"Data ingestion successful. File saved to: {output_path}")
    else:
        raise FileNotFoundError("Data download failed.")

    return output_path


if __name__ == '__main__':
    config = load_config()
    ingest_data(config)
