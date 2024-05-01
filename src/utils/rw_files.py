"""
Utils functions to read and write files
"""

import os
import csv
import logging
import pandas as pd
import geopandas as gpd
import zipfile
import requests
from typing import Optional

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


def check_shp_extension(filename: str) -> bool:
    """
    Check if filename ends with .shp
    """
    return filename.endswith(".shp")


def check_directory_exists(directory: str) -> bool:
    """
    Check if directory exists.
    """
    return os.path.exists(directory)


def detect_delimiter(filename: str) -> str:
    """
    Detects the delimiter of a CSV file.

    This function opens a file for reading and uses the csv.Sniffer class to infer the delimiter
    used in the CSV file. It reads the first 1024 bytes of the file for detection, which is usually
    sufficient for correctly identifying the delimiter in well-formatted CSV files.

    Parameters:
    - filename (str): The path to the CSV file.

    Returns:
    - str: The detected delimiter of the file. Common delimiters include commas (','), tabs ('\t'),
      and semicolons (';'). If the delimiter cannot be determined, the function returns None.
    """
    with open(filename, "r") as file:
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(file.read(1024))
            return dialect.delimiter
        except csv.Error as e:
            logging.warning("Could not determine the delimiter.")
            return None


def load_data(filename: str) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Load data from CSV or pickle file.
    """
    if check_csv_extension(filename):
        return load_csv(filename)
    if check_pkl_extension(filename):
        return load_pickle(filename)
    if check_shp_extension(filename):
        return load_shape(filename)

    logging.error("File extension not supported.")
    raise ValueError("File extension not supported.")


def load_csv(filename: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    """
    logging.info("Loading CSV data %s", filename)
    delimiter = detect_delimiter(filename)
    try:
        return pd.read_csv(filename, sep=delimiter)
    except pd.errors.ParserError as e:
        logging.error(e)
        return pd.read_csv(filename, sep=None, engine="python")


def load_pickle(filename: str) -> pd.DataFrame:
    """
    Load data from pickle file.
    """
    logging.info("Loading pickle data %s", filename)
    return pd.read_pickle(filename)


def load_shape(filename: str) -> gpd.GeoDataFrame:
    """
    Load data from shape file.
    """
    logging.info("Loading shape data %s", filename)
    return gpd.read_file(filename)


def save_data(
    df: pd.DataFrame,
    filename: str,
    save_csv: bool = True,
    save_pickle: bool = True,
    index: bool = False,
) -> None:
    """
    Save dataframe as CSV or pickle file.
    """
    logging.info("Saving data.")
    if save_csv:
        save_csv_data(df, filename, index=index)
    if save_pickle:
        save_pickle_data(df, filename)


def save_csv_data(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """
    Save dataframe as CSV file.
    """
    logging.info("Saving data as CSV.")

    # Check if filename ends with .csv
    if not filename.endswith(".csv"):
        filename = filename + ".csv"

    try:
        df.to_csv(filename, index=index)
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


def download_file(url: str, save_path: str) -> None:
    """
    Downloads a file from a given URL and saves it to a specified path.

    Parameters:
    - url (str): The URL of the file to download.
    - save_path (str): The full path to save the file to.
    """
    try:
        response = requests.get(url, timeout=20)  # Add timeout argument
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(save_path, "wb") as file:
            file.write(response.content)
        logging.info("Downloaded file saved to %s", save_path)
    except requests.RequestException as e:
        logging.error("Error downloading file: %s", e)


def unzip_file(zip_path: str, extract_to: Optional[str]) -> None:
    """
    Extracts a ZIP file to a specified directory.

    Parameters:
    - zip_path (str): The path of the ZIP file to extract.
    - extract_to (str, optional): The directory to extract the contents to.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info("Extracted %s to %s", os.path.basename(zip_path), extract_to)
    except zipfile.BadZipFile as e:
        logging.error("Error extracting the ZIP file: %s", e)
