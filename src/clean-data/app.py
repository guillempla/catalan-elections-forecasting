import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_catalan_elections_data():
    """
    Load data from CSV file.
    """
    logging.info("Loading data from CSV.")
    return pd.read_csv("data/raw/catalan-elections-data.csv")


def load_election_days_data():
    """
    Load data from CSV file.
    """
    logging.info("Loading data from CSV.")
    return pd.read_csv("data/processed/election-days.csv")


def drop_columns(df, columns_to_drop: list):
    """
    Drop columns from dataframe.
    """
    logging.info("Dropping columns: " + columns_to_drop)
    return df.drop(columns=columns_to_drop)


def rename_columns(df, columns_to_rename: dict):
    """
    Rename columns from dataframe.
    """
    logging.info("Renaming columns: " + columns_to_rename)
    return df.rename(columns=columns_to_rename)


def divide_id_eleccio(df):
    """
    Divide id_eleccio column into id_partit and id_circunscripcio.
    """
    logging.info("Dividing id_eleccio column.")
    df["type"] = df["id_eleccio"].str[:1]
    df["year"] = df["id_eleccio"].str[1:5].astype(int)
    df["sequential"] = df["id_eleccio"].str[5:]
    return df


def merge_election_days(df_elections_data, df_elections_days):
    """
    Merge election_days dataframe into df.
    """
    logging.info("Merging election_days dataframe.")
    return df_elections_data.merge(
        df_elections_days, on=["nom_eleccio", "year"], how="left"
    )


def filter_by_election_type(df, election_type: list):
    """
    Filter dataframe by election type.
    """
    logging.info("Filtering by election type.")

    # Check if election_type is a list
    if not isinstance(election_type, list):
        raise TypeError("election_type must be a list.")

    # Check if election_type is empty
    if not election_type:
        raise ValueError("election_type cannot be empty.")

    # Check if election_type contains valid values
    valid_election_type = ["A", "C", "D", "E", "G", "M", "S", "V"]
    if not all(elem in valid_election_type for elem in election_type):
        raise ValueError("election_type contains invalid values.")

    return df[df["type"].isin(election_type)]


def save_data(df, filename: str):
    """
    Save dataframe as CSV file.
    """
    logging.info("Saving data as CSV.")
    df.to_csv(filename, index=False)
