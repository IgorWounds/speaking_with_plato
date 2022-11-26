import os
import logging
from typing import Literal


def download_model(model_path: str, model_name: Literal["chatbot", "writebot"]) -> None:
    """
    Download the model from Google Drive.
    """
    os.chdir(model_path)
    logging.info(f"Downloading {model_name} model...")
    if model_name == "writebot":
        url = "15aQCUMY_UAD3bikl1MARxi7j4dTNY4Ia"
        file = "GPT-2.zip"
    elif model_name == "chatbot":
        url = "1QLEF2KVXKvfroAmqrNQK4Q6G8qFcgSmG"
        file = "dialouGPT.zip"
    else:
        raise ValueError("model_name must be either 'writebot' or 'chatbot'")

    os.system(f"gdown {url} -O {model_path + file} | tar -xvzf {file}")
