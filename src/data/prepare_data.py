import os
import re
import json
import shutil
import logging

from zipfile import ZipFile
import html2text

import pandas as pd


def extract_files(zip: str, files: str) -> None:
    # extract files frrom zip by using the extract_files.sh script
    logging.info("Extracting files from zip")
    with ZipFile(zip, "r") as zipObj:
        zipObj.extractall(files)
    logging.info("Done")


def move_files(folder: str, new_folder: str) -> None:
    files = os.listdir(folder + "/29441-h/files")
    logging.info(f"Moving {len(files)} files from {folder} to {new_folder}")
    for file in files:
        # we don't want the 13726-h file as it combines other three files
        if "13726" not in file and "29441" not in file:
            try:
                shutil.move(
                    (folder + "/29441-h/files" + "/" + file + "/" + file + ".htm"),
                    new_folder + "/" + file,
                )
            except FileNotFoundError:
                pass
    logging.info("Deleting text parent folder")
    shutil.rmtree(folder + "/29441-h")
    logging.info("Done")


def rename_files(folder: str) -> None:
    """Rename the files to the name of the original work"""
    files = os.listdir(folder)
    file_names = json.load(open("speaking_with_plato/src/data/file_names.json"))
    logging.info(f"Renaming {len(files)} files from {folder}")
    for file in files:
        if file in file_names.keys():
            os.rename(
                os.path.join(folder, file),
                os.path.join(folder, file_names[file].lower()),
            )
    logging.info("Done")


def clean_files(folder: str, move_to: str) -> None:
    """Clean the files and save them in the interim folder"""

    h = html2text.HTML2Text()
    files = os.listdir(folder)
    logging.info(f"Cleaning {len(files)} files from {folder}")
    for file in files:
        text = h.handle(open(os.path.join(folder, file)).read())
        text = re.sub(r".*<pre>", "", text)
        text = re.sub(r"</pre>.*", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"&.*?;", "", text)
        text = re.sub(r"\\n", "", text)
        text = re.sub(r"\\t", "", text)
        text = re.sub(r"\\r", "", text)
        text = re.sub(r"\\", "", text)
        text = re.sub(r"\\xa0", "", text)
        text = re.sub(r"\\x9d", "", text)
        text = re.sub(r"\\x9c", "", text)
        text = re.sub(r"\\x8c", "", text)
        text = re.sub(r"\\x8d", "", text)
        text = re.sub(r"\\x8e", "", text)
        text = re.sub(r"\\x8f", "", text)
        text = re.sub(r"\\x90", "", text)
        text = re.sub(r"\\x91", "", text)
        text = re.sub(r"\\x92", "", text)
        text = re.sub(r"\\x93", "", text)
        text = re.sub(r"\\x94", "", text)
        text = re.sub(r"\\x95", "", text)
        text = re.sub(r"\\x96", "", text)
        text = re.sub(r"\\x97", "", text)
        text = re.sub(r"\\x98", "", text)
        text = re.sub(r"\\x99", "", text)
        text = re.sub(r"\\x9a", "", text)
        text = re.sub(r"\\x9b", "", text)
        text = re.sub(r"\\x9c", "", text)
        text = re.sub(r"\\x9d", "", text)
        text = re.sub(r"\\x9e", "", text)
        text = re.sub(r"\\x9f", "", text)
        text = re.sub(r"\\xa0", "", text)
        text = re.sub(r"\\n\\n", "", text)
        text = re.sub(r"\\xa0\\xa0\\xa0\\xa0.*", "", text)
        text = re.sub(r".*" + file.upper(), file.upper(), text)
        text = re.sub(r".*\*\* START", "", text)
        text = re.sub(r"END OF THE PROJECT GUTENBERG EBOOK.*", "", text)
        text = re.sub(r"End of the Project Gutenberg EBook.*", "", text)
        text = re.sub(r"END OF THIS PROJECT GUTENBERG EBOOK ALCIBIADES I.*", "", text)
        text = re.sub(r"\\n\\n", "", text)
        text = re.sub(r"\*\*\*\*\*.*", "", text)
        text = re.sub(r".*TRANSLATED BY.*", "", text)
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"\(|\)", "", text)

        with open(os.path.join(move_to, file), "w") as f:
            f.write(text)

    shutil.rmtree(folder)
    logging.info("Done")


def csv_clean(text: str) -> str:
    """Clean the text some more prior to creating the csv that will be used for EDA"""
    # text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text


def create_csv(folder: str, save_path: str) -> None:
    """Create a csv file for EDA where each column is Plato's work"""
    logging.info("Saving csv file")
    data = pd.DataFrame()
    for root, _, files in os.walk(folder):
        files.remove(".gitkeep")
        for file in files:
            with open(os.path.join(root, file), "r") as f:
                text = f.read()
                text = csv_clean(text)
                data[file] = [text]
    data.to_csv(save_path + "/plato_texts.csv", index=False)
    logging.info("Done")


def clean_dialogue(text: str) -> str:
    # clear all new lines
    text = re.sub(r"\n", " ", text)
    # add a new line when an UPPERCASE word with a semi colon is found
    text = re.sub(r"([A-Z]+:)", r"\n\1", text)
    # make all whole UPPERCASE words start with an uppercase letter and the rest lowercase
    text = re.sub(r"([A-Z]+)", lambda x: x.group(1).lower().capitalize(), text)
    return text


def combine_and_create_texts(folder: str, save_path: str) -> None:
    """Combines all the texts into a single file for deep learning text generation
    and create a file only with dialogues that fit the Q&A format"""
    not_dialogue = ["apology", "laws"]
    unfit_dialogue = [
        "parmenides",
        "symposium",
        "lysis",
        "the_republic",
        "charmides",
    ]
    logging.info("Combining all files and creating new ones for generation and Q&A")
    files = os.listdir(folder)
    for file in files:
        with open(os.path.join(folder, file), "r") as f:
            text = f.read()
            if file not in not_dialogue and file not in unfit_dialogue:
                text = clean_dialogue(text)
                with open(os.path.join(save_path, "dialogues.txt"), "a") as f:
                    f.write(text)
            else:
                text = re.sub(r"\n", " ", text)

            with open(os.path.join(save_path, "all_plato.txt"), "a") as f:
                f.write(text)
    logging.info("Done")
