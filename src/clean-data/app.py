"""
Clean data and save it as a CSV file.
"""
from typing import List, Dict

import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_csv(filename: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    """
    logging.info("Loading data from CSV.")
    return pd.read_csv(filename)


def drop_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drop columns from dataframe.
    """
    logging.info("Dropping columns: %s", columns_to_drop)
    return df.drop(columns=columns_to_drop)


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
    df["sequential"] = df["id_eleccio"].str[5:]
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


def save_data(df: pd.DataFrame, filename: str):
    """
    Save dataframe as CSV file.
    """
    logging.info("Saving data as CSV.")
    df.to_csv(filename, index=False)


class CleanData:
    """
    Clean data.
    """

    def __init__(
        self,
        elections_data_filename: str = "data/raw/catalan-elections-data.csv",
        elections_days_filename: str = "data/processed/election-days.csv",
        output_filename: str = "data/processed/catalan-elections-clean-data.csv",
        elections_type: List[str] = [
            "M",
            "E",
            "A",
            "G",
        ],  # M: Municipals, E: Europees, A: Autonòmiques, G: Generals
    ) -> None:
        """
        Initialize class.
        """
        self.df = load_csv(elections_data_filename)
        self.elections_days_df = load_csv(elections_days_filename)
        self.output_filename = output_filename
        self.elections_type = elections_type

    def clean_elections_data(self):
        """
        Clean elections data.
        """
        logging.info("Cleaning elections data.")
        self.df = (
            self.df.pipe(drop_columns, columns_to_drop=["candidat_posicio"])
            .pipe(rename_columns, columns_to_rename={"secci_": "seccio"})
            .pipe(divide_id_eleccio)
            .pipe(filter_by_election_type, elections_type=self.elections_type)
            .pipe(merge_election_days, df_elections_days=self.elections_days_df)
            .pipe(save_data, filename=self.output_filename)
        )
