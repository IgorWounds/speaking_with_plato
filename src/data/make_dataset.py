# -*- coding: utf-8 -*-
import logging
import shutil
from prepare_data import (
    extract_files,
    move_files,
    rename_files,
    clean_files,
    create_csv,
    combine_and_create_texts,
)

ZIP_PATH = "speaking_with_plato/data/external/29441-h.zip"
EXTERNAL_FILES = "speaking_with_plato/data/external"
RAW_FILES = "speaking_with_plato/data/raw"
INTERIM_FILES = "speaking_with_plato/data/interim"
PROCESSED_FILES = "speaking_with_plato/data/processed"


def main():
    """Extracts Plato's works from zip.
    Runs data processing scripts to turn raw data from (../raw) into
    semi-cleaned data ready to be manually cleaned from license and work interpretation (saved in ../interim).
    A prompt will wait for when the data is manually cleaned.
    After the user writes "Yes", the data is cleaned further and files are generated and saved in ../processed .
    Saved files are a csv used for EDA and two txt files for modeling.
    """
    logger = logging.getLogger(__name__)
    logger.info("Preparing data for modeling.")

    extract_files(zip=ZIP_PATH, files=EXTERNAL_FILES)
    move_files(folder=EXTERNAL_FILES, new_folder=RAW_FILES)
    rename_files(folder=RAW_FILES)
    clean_files(folder=RAW_FILES, move_to=INTERIM_FILES)

    logger.info("Data is ready to be manually cleaned.")

    if input("Is the data completely cleaned? (Yes/No)") == "Yes":
        logger.info("Finalizing data cleaning and generating files.")
        create_csv(folder=INTERIM_FILES, save_path=PROCESSED_FILES)
        combine_and_create_texts(folder=INTERIM_FILES, save_path=PROCESSED_FILES)
        shutil.make_archive(INTERIM_FILES, "zip", INTERIM_FILES)
        logger.info("Done")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
