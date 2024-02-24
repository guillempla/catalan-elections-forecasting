"""
Utils functions to read and write files
"""

import logging
import os
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def fix_file_extension(filename: str, extension: str) -> str:
    """
    Check if filename ends with a specific extension,
    if not the extension is added to the filename.
    """
    if not filename.endswith(extension):
        filename = filename + extension
    return filename


def check_file_exists(filename: str) -> bool:
    """
    Check if file exists.
    """
    try:
        with open(filename, "r", encoding="utf-8"):
            return True
    except FileNotFoundError:
        return False


def check_pkl_extension(filename: str) -> bool:
    """
    Check if filename ends with .pkl
    """
    return filename.endswith(".pkl")


def check_csv_extension(filename: str) -> bool:
    """
    Check if filename ends with .csv
    """
    return filename.endswith(".csv")


def check_directory_exists(directory: str) -> bool:
    """
    Check if directory exists.
    """
    return os.path.exists(directory)


def load_data(filename: str) -> pd.DataFrame:
    """
    Load data from CSV or pickle file.
    """
    if check_csv_extension(filename):
        return load_csv(filename)
    elif check_pkl_extension(filename):
        return load_pickle(filename)
    else:
        logging.error("File extension not supported.")
        raise ValueError("File extension not supported.")


def load_csv(filename: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    """
    logging.info("Loading data from CSV.")
    return pd.read_csv(filename)


def load_pickle(filename: str) -> pd.DataFrame:
    """
    Load data from pickle file.
    """
    logging.info("Loading data from pickle.")
    return pd.read_pickle(filename)


def save_data(
    df: pd.DataFrame, filename: str, save_csv: bool = True, save_pickle: bool = True
) -> None:
    """
    Save dataframe as CSV or pickle file.
    """
    logging.info("Saving data.")
    if save_csv:
        save_csv_data(df, filename)
    if save_pickle:
        save_pickle_data(df, filename)


def save_csv_data(df: pd.DataFrame, filename: str):
    """
    Save dataframe as CSV file.
    """
    logging.info("Saving data as CSV.")

    # Check if filename ends with .csv
    if not filename.endswith(".csv"):
        filename = filename + ".csv"

    try:
        df.to_csv(filename, index=False)
    except (FileNotFoundError, PermissionError) as e:
        logging.error(e)


def save_pickle_data(df: pd.DataFrame, filename: str):
    """
    Save dataframe as pickle file.
    """
    logging.info("Saving data as pickle.")

    # Check if filename ends with .pkl
    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"

    try:
        df.to_pickle(filename)
    except (FileNotFoundError, PermissionError) as e:
        logging.error(e)
