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


def create_party_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create party columns.
    """
    logging.info("Creating party columns.")
    df.loc[df["agrupacio_codi"].notnull(), "party_code"] = df["agrupacio_codi"]
    df.loc[df["agrupacio_denominacio"].notnull(), "party_name"] = df[
        "agrupacio_denominacio"
    ]
    df.loc[df["agrupacio_sigles"].notnull(), "party_abbr"] = df["agrupacio_sigles"]
    df["party_color"] = df["candidatura_color"]
    return df


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


def save_data(
    df: pd.DataFrame, filename: str, save_csv: bool = True, save_picke: bool = True
) -> None:
    """
    Save dataframe as CSV or pickle file.
    """
    logging.info("Saving data.")
    if save_csv:
        save_csv_data(df, filename)
    if save_picke:
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
            .pipe(save_data, filename=self.output_filename)
        )
