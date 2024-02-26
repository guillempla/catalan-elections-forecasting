"""
Clean data and save it as a CSV file.
"""

from typing import List, Dict

import logging
import pandas as pd
import numpy as np
from unidecode import unidecode
from utils.rw_files import load_data, save_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def clean_party_name(party_name: str) -> str:
    """
    Cleans the given party name by converting it to lowercase, removing diacritics,
    and removing parentheses, commas, and hyphens.

    Args:
        party_name (str): The party name to be cleaned.

    Returns:
        str: The cleaned party name.
    """
    party_name = party_name.lower()
    party_name = unidecode(party_name)
    party_name = (
        party_name.replace("(", "").replace(")", "").replace(",", "").replace("-", "")
    )
    return party_name


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
    df["clean_party_name"] = df["party_name"].apply(clean_party_name)
    df["clean_party_abbr"] = df["party_abbr"].apply(clean_party_name)
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


def replace_rows_with_null_values(
    df: pd.DataFrame, empty_columns: List[str], value: str
) -> pd.DataFrame:
    """
    Replace rows with null values in specified columns.
    """
    logging.info("Replacing rows with null values in columns: %s", empty_columns)
    df[empty_columns] = df[empty_columns].fillna(value)
    return df


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
        clean_configs: List[Dict],
    ) -> None:
        """
        Initialize class.
        """

        for config in clean_configs:
            self.elections_data_filename = config.get(["elections_data_filename"])
            self.elections_days_filename = config.get(["elections_days_filename"])

            if self.elections_data_filename is None:
                raise ValueError("Elections data filename cannot be empty.")
            self.df = load_data(self.elections_data_filename)

            self.elections_days_df = None
            if self.elections_days_filename is not None:
                self.elections_days_df = load_data(self.elections_days_filename)

            self.output_filename = config.get("output_filename")
            self.columns_to_drop = config.get("columns_to_drop")
            self.columns_to_rename = config.get("columns_to_rename")
            self.elections_type = config.get("elections_type")
            self.color_column = config.get("color_column")
            self.color_default = config.get("color_default")
            self.columns_types = config.get("columns_types")
            self.columns_null_values = config.get("columns_null_values")

            self.run_columns_null_values = self.columns_null_values is not None
            self.run_create_party_column = config.get("create_party_column")
            self.run_divide_id_eleccio = config.get("divide_id_eleccio")
            self.run_rename_columns = self.columns_to_rename is not None
            self.run_filter_by_election_type = self.elections_type is not None
            self.run_merge_election_days = self.elections_days_df is not None
            self.run_create_date_column = config.get("create_data_column")
            self.run_replace_nan_colors = self.color_column is not None
            self.run_drop_columns = self.columns_to_drop is not None
            self.run_set_column_type = self.columns_types is not None

            self.clean_elections_data()

    def clean_elections_data(self):
        """
        Clean elections data.
        """
        logging.info(f"Cleaning elections data.")
        if self.run_columns_null_values:
            self.df = replace_rows_with_null_values(
                self.df, empty_columns=self.columns_null_values, value=""
            )
        if self.run_create_party_column:
            self.df = create_party_columns(self.df)
        if self.run_divide_id_eleccio:
            self.df = divide_id_eleccio(self.df)
        if self.run_rename_columns:
            self.df = rename_columns(self.df, columns_to_rename=self.columns_to_rename)
        if self.run_filter_by_election_type:
            self.df = filter_by_election_type(
                self.df, elections_type=self.elections_type
            )
        if self.run_merge_election_days:
            self.df = merge_election_days(
                self.df, df_elections_days=self.elections_days_df
            )
        if self.run_create_date_column:
            self.df = create_date_column(self.df)
        if self.run_replace_nan_colors:
            self.df = replace_nan_colors(
                self.df, column=self.color_column, color=self.color_default
            )
        if self.run_drop_columns:
            self.df = drop_columns(self.df, columns_to_drop=self.columns_to_drop)
        if self.run_set_column_type:
            self.df = set_column_type(self.df, column_types=self.columns_types)

        save_data(self.df, self.output_filename)
