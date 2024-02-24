"""
Clean data and save it as a CSV file.
"""

from typing import List, Dict

import logging
import pandas as pd
import numpy as np
from utils.rw_files import load_csv, save_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def create_party_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create party columns.
    """
    logging.info("Creating party columns.")
    df["party_code"] = np.where(
        df["agrupacio_codi"].notnull(), df["agrupacio_codi"], df["candidatura_codi"]
    )
    df["party_name"] = np.where(
        df["agrupacio_denominacio"].notnull(),
        df["agrupacio_denominacio"],
        df["candidatura_denominacio"],
    )
    df["party_abbr"] = np.where(
        df["agrupacio_sigles"].notnull(),
        df["agrupacio_sigles"],
        df["candidatura_sigles"],
    )
    df["party_color"] = df["candidatura_color"]
    return df


def drop_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drop columns from dataframe.
    """
    logging.info("Dropping columns: %s", columns_to_drop)
    return df.drop(columns=columns_to_drop)


def remove_rows_with_null_values(
    df: pd.DataFrame, empty_columns: List[str]
) -> pd.DataFrame:
    """
    Remove rows with null values in specified columns.
    """
    logging.info("Removing rows with null values in columns: %s", empty_columns)
    return df.dropna(subset=empty_columns)


def rename_columns(df: pd.DataFrame, columns_to_rename: Dict[str, str]) -> pd.DataFrame:
    """
    Rename columns from dataframe.
    """
    logging.info("Renaming columns: %s", columns_to_rename)
    return df.rename(columns=columns_to_rename)


def divide_id_eleccio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Divide id_eleccio column into id_partit and id_circunscripcio.
    """
    logging.info("Dividing id_eleccio column.")
    df["type"] = df["id_eleccio"].str[:1]
    df["year"] = df["id_eleccio"].str[1:5].astype(int)
    df["round"] = df["id_eleccio"].str[5:]
    return df


def merge_election_days(
    df_elections_data: pd.DataFrame, df_elections_days: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge election_days dataframe into df.
    """
    logging.info("Merging election_days dataframe.")
    return df_elections_data.merge(
        df_elections_days, on=["nom_eleccio", "year"], how="left"
    )


def filter_by_election_type(
    df: pd.DataFrame, elections_type: List[str]
) -> pd.DataFrame:
    """
    Filter dataframe by election type.
    """
    logging.info("Filtering by election type.")

    # Check if election_type is a list
    if not isinstance(elections_type, list):
        raise TypeError("election_type must be a list.")

    # Check if election_type is empty
    if not elections_type:
        raise ValueError("election_type cannot be empty.")

    # Check if election_type contains valid values
    valid_election_type = df["type"].unique()
    if not all(elem in valid_election_type for elem in elections_type):
        raise ValueError("election_type contains invalid values.")

    return df[df["type"].isin(elections_type)]


def create_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create date column.
    """
    logging.info("Creating date column.")
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    return df


def replace_nan_colors(
    df: pd.DataFrame, column="candidatura_color", color="grey"
) -> pd.DataFrame:
    """
    Replace NaN values in colors columns.
    """
    logging.info("Replacing NaN values in colors columns.")
    df[column].fillna(color, inplace=True)
    return df


def set_column_type(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """
    Set column types according to a dictionary mapping column names to types,
    with support for nullable integer types to handle NaN values.

    Parameters:
    - df: pandas DataFrame.
    - column_types: Dictionary where keys are column names and values are the data types.

    Returns:
    - DataFrame with updated column types.
    """
    logging.info("Setting column type.")

    for column, dtype in column_types.items():
        if column in df.columns:
            # Automatically use nullable integer types if dtype is 'int'
            if dtype == "int" and df[column].isnull().any():
                dtype = "Int64"
            try:
                df[column] = df[column].astype(dtype)
            except Exception as e:
                logging.error(e)
                logging.error("Error setting column type for %s.", column)
            logging.info("Column %s type set to %s.", column, dtype)
        else:
            logging.warning("Column %s not found in DataFrame.", column)
    return df


class CleanData:
    """
    Clean data.
    """

    def __init__(
        self,
        elections_data_filename: str = "../data/raw/catalan-elections-data.csv",
        elections_days_filename: str = "../data/processed/elections_days.csv",
        output_filename: str = "../data/processed/catalan-elections-clean-data",
        columns_to_drop: List[str] = [
            "candidat_posicio",
            "candidatura_logotip",
            "id_eleccio",
            "candidatura_codi",
            "candidatura_denominacio",
            "candidatura_sigles",
            "candidatura_color",
            "agrupacio_codi",
            "agrupacio_denominacio",
            "agrupacio_sigles",
        ],
        columns_to_rename: Dict[str, str] = {"secci_": "seccio"},
        elections_type: List[str] = [
            "M",
            "E",
            "A",
            "G",
        ],  # M: Municipals, E: Europees, A: Autonòmiques, G: Generals
        color_column: str = "candidatura_color",
        color_default: str = "grey",
        columns_types: dict = {
            "year": "int",
            "month": "int",
            "day": "int",
            "seccio": "int",
            "vots": "int",
            "escons": "int",
            "districte": "int",
            "party_code": "int",
        },
        columns_null_values: List[str] = ["candidatura_sigles"],
    ) -> None:
        """
        Initialize class.
        """
        self.df = load_csv(elections_data_filename)
        self.elections_days_df = load_csv(elections_days_filename)
        self.output_filename = output_filename
        self.columns_to_drop = columns_to_drop
        self.columns_to_rename = columns_to_rename
        self.elections_type = elections_type
        self.color_column = color_column
        self.color_default = color_default
        self.columns_types = columns_types
        self.columns_null_values = columns_null_values

    def clean_elections_data(self):
        """
        Clean elections data.
        """
        logging.info("Cleaning elections data.")
        self.df = (
            self.df.pipe(create_party_columns)
            .pipe(divide_id_eleccio)
            .pipe(rename_columns, columns_to_rename=self.columns_to_rename)
            .pipe(filter_by_election_type, elections_type=self.elections_type)
            .pipe(merge_election_days, df_elections_days=self.elections_days_df)
            .pipe(create_date_column)
            .pipe(
                replace_nan_colors, column=self.color_column, color=self.color_default
            )
            .pipe(drop_columns, columns_to_drop=self.columns_to_drop)
            .pipe(remove_rows_with_null_values, empty_columns=self.columns_null_values)
            .pipe(set_column_type, column_types=self.columns_types)
            .pipe(save_data, filename=self.output_filename)
        )
