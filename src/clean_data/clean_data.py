"""
Clean data and save it as a CSV file.
"""

import os
import re
from typing import Any, List, Dict

import logging
import pandas as pd
import numpy as np
from unidecode import unidecode
from utils.municipal_code_control_digit import MunicipalCodeControlDigit
from utils.rw_files import load_data, save_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - Clean Data - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def aggregate_duplicated_parties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates duplicated parties in the given DataFrame.

    For each duplicated party on the same election on the same census section,
    they are aggregated into a single row by summing the votes and seats.

    Args:
        df (pd.DataFrame): The DataFrame containing the party data.

    Returns:
        pd.DataFrame: The DataFrame with duplicated parties aggregated.

    """
    logging.info("Aggregating duplicated parties.")

    # Identifying duplicates based on election, census section, and party code
    duplicates_mask = df.duplicated(
        subset=["mundissec", "party_code", "id_eleccio"], keep=False
    )
    duplicated_df = df[duplicates_mask]

    # Define how each column should be aggregated
    aggregations = {
        "vots": "sum",
        "escons": "first",
        "vots_valids": "first",
        "vots_blancs": "first",
        "vots_nuls": "first",
        "votants": "first",
        "cens_electoral": "first",
        "party_name": "first",
        "party_abbr": "first",
        "party_color": "first",
        "type": "first",
        "year": "first",
        "round": "first",
        "clean_party_name": "first",
        "clean_party_abbr": "first",
        "date": "first",
    }

    # Aggregate duplicated rows
    aggregated_df = (
        duplicated_df.groupby(["mundissec", "party_code", "id_eleccio"])
        .agg(aggregations)
        .reset_index()
    )

    # Recalculate percentage columns
    for col in [
        "vots_valids_percentage",
        "cens_electoral_percentage",
        "votants_percentage",
    ]:
        aggregated_df[col] = (
            aggregated_df["vots"] / aggregated_df[col.replace("_percentage", "")] * 100
        )

    # Merge the aggregated results back to the original dataframe
    # First, drop duplicates from the original df
    df_cleaned = df.drop(index=duplicated_df.index)

    # Append the aggregated data
    final_df = pd.concat([df_cleaned, aggregated_df], ignore_index=True)

    return final_df


def fix_party_codes(df: pd.DataFrame, party_codes_to_fix: Dict) -> pd.DataFrame:
    """
    Fix party codes.
    """
    logging.info("Fixing party codes.")
    df1 = df.copy()
    for party_code, party_columns in party_codes_to_fix.items():
        party_code_int = int(party_code)
        new_code = int(party_code + "1")
        code_column = party_columns.get("code_column")
        name_column = party_columns.get("name_column")

        # Select rows to update based on the unique name_column values associated with party_code
        unique_names = df1[df1[code_column] == party_code_int][name_column].unique()
        for name in unique_names:
            df1.loc[
                (df1[code_column] == party_code_int) & (df1[name_column] == name),
                code_column,
            ] = new_code
            new_code += 1

    return df1


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
        (df["agrupacio_codi"].notnull() & (df["agrupacio_codi"] != "")),
        df["agrupacio_codi"],
        df["candidatura_codi"],
    )
    # Convert party_code to integer if not empty, handle empty strings or potential nulls after where condition
    df["party_code"] = (
        df["party_code"].replace("", np.nan).astype(float).astype("Int64")
    )

    df["party_name"] = np.where(
        (df["agrupacio_denominacio"].notnull() & (df["agrupacio_denominacio"] != "")),
        df["agrupacio_denominacio"],
        df["candidatura_denominacio"],
    )

    df["party_abbr"] = np.where(
        (df["agrupacio_sigles"].notnull() & (df["agrupacio_sigles"] != "")),
        df["agrupacio_sigles"],
        df["candidatura_sigles"],
    )
    df["party_color"] = df["candidatura_color"]
    df["clean_party_name"] = df["party_name"].apply(clean_party_name)
    df["clean_party_abbr"] = df["party_abbr"].apply(clean_party_name)
    return df


def create_mundissec_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create mundissec column.
    """
    logging.info("Creating mundissec columns.")

    # Initialize the mundissec column with None
    df["mundissec"] = None

    # Filter rows where id_nivell_territorial is in the specified list
    filter_rows = df["id_nivell_territorial"].isin(["DM", "ME", "MU", "SE"])
    # Calculate control code only for the filtered rows
    control_code = (
        df.loc[filter_rows, "territori_codi"]
        .astype(int)
        .apply(MunicipalCodeControlDigit.calculate)
        .astype(str)
    )

    # Update the calculation of mundissec only for rows that meet the condition
    territori_codi = (
        df.loc[filter_rows, "territori_codi"].astype(str) + control_code
    ).str.zfill(5)
    districte = df.loc[filter_rows, "districte"].astype(str).str.zfill(2)
    seccio = df.loc[filter_rows, "seccio"].astype(str).str.zfill(3)
    df.loc[filter_rows, "mundissec"] = territori_codi + districte + seccio
    return df


def drop_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drop columns from dataframe.
    """
    logging.info("Dropping columns: %s", columns_to_drop)
    return df.drop(columns=columns_to_drop)


def remove_empty_rows(df: pd.DataFrame, empty_columns: List[str]) -> pd.DataFrame:
    """
    Remove rows with null values in specified columns.
    """
    logging.info("Removing rows with null values in columns: %s", empty_columns)
    return df.dropna(subset=empty_columns)


def replace_rows_values(df: pd.DataFrame, column: str, old_value: str, new_value: str):
    """
    Replace rows with old_value in column with new_value.
    """
    logging.info(
        "Replacing rows with %s in column %s with %s.", old_value, column, new_value
    )
    df[column] = df[column].replace(old_value, new_value)
    return df


def remove_rows_by_values(df: pd.DataFrame, conditions: list):
    """
    Remove rows based on multiple conditions.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        conditions (list): A list of dictionaries, each containing a 'column' key and 'value' key,
                           specifying which rows to remove from the DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with the specified rows removed.
    """
    for condition in conditions:
        column = condition["column"]
        value = condition["value"]
        logging.info("Removing rows with %s in column %s.", value, column)
        df = df[df[column] != value]
    return df


def fix_mundissec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the mundissec code as some of the codes have invalid (or none) check codes.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: The modified DataFrame with the fixed mundissec codes.
    """
    mundissec_codes = df["mundissec"]
    municipal_codes = mundissec_codes.apply(lambda x: x[:5])
    check_codes = (
        municipal_codes.astype(int)
        .apply(MunicipalCodeControlDigit.calculate)
        .astype(str)
    )
    rest_of_code = mundissec_codes.apply(lambda x: x[5:])
    fixed_mundissec = municipal_codes + check_codes + rest_of_code

    df["mundissec"] = fixed_mundissec
    return df


def calculate_p_born_abroad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the proportion of people born abroad in the population.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: The modified DataFrame with the new column 'p_born_abroad' added.
    """
    logging.info("Calculating the proportion of people born abroad in the population.")
    df = df.copy()
    # Pivot the data to get 'Nacidos en el Extranjero' and 'Total Población' as columns
    pivot_df = df.pivot_table(
        index=["mundissec", "year"],
        columns="País de nacimiento",
        values="Total",
        aggfunc="sum",
    )

    # Calculate the proportion of 'Nacidos en el Extranjero' to 'Total Población'
    pivot_df["p_born_abroad"] = (
        pivot_df["Nacidos en el Extranjero"] / pivot_df["Total Población"]
    )

    # Reset the index if needed to flatten the DataFrame
    pivot_df.reset_index(inplace=True)

    return pivot_df


def replace_empty_rows(
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


def filter_by_column(df: pd.DataFrame, column: str, values: List[str]) -> pd.DataFrame:
    """
    Filter dataframe by column.
    """
    logging.info("Filtering by column %s.", column)

    # Check if column is in DataFrame
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame.")

    # Check if values is a list
    if not isinstance(values, list):
        raise TypeError("values must be a list.")

    # Check if values is empty
    if not values:
        raise ValueError("values cannot be empty.")

    # Check if values contains valid values
    valid_values = df[column].unique()
    if not all(elem in valid_values for elem in values):
        raise ValueError("invalid values.")

    return df[df[column].isin(values)]


def create_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create date column.
    """
    logging.info("Creating date column.")
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    return df


def create_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create date columns.
    """
    logging.info("Creating date columns.")
    df["data_eleccio"] = pd.to_datetime(
        df["data_eleccio"]
    )  # Convert data_eleccio to datetime
    df["year"] = df["data_eleccio"].dt.year
    df["month"] = df["data_eleccio"].dt.month
    df["day"] = df["data_eleccio"].dt.day
    return df


def replace_nan_colors(
    df: pd.DataFrame, column="candidatura_color", color="grey"
) -> pd.DataFrame:
    """
    Replace NaN values in colors columns.
    """
    logging.info("Replacing NaN values in colors columns.")
    df[column] = df[column].replace(np.nan, color)
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


def merge_participation_data(
    df_votes: pd.DataFrame, df_participation: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge participation data into df.
    """
    logging.info("Merging participation data.")

    # Filter out rows with null values in id_eleccio, territori_codi, districte, seccio
    # to avoid problems on merge
    if (
        df_votes[["id_eleccio", "territori_codi", "districte", "seccio"]]
        .isnull()
        .values.any()
    ):
        logging.warning(
            "Null values found in df_votes' columns: id_eleccio, territori_codi, districte, seccio."
        )
        df_votes = df_votes.dropna(
            subset=["id_eleccio", "territori_codi", "districte", "seccio"]
        )
    if (
        df_participation[["id_eleccio", "territori_codi", "districte", "seccio"]]
        .isnull()
        .values.any()
    ):
        logging.warning(
            "Null values found in df_participation's columns: id_eleccio, territori_codi, districte, seccio in df_participation."
        )
        df_participation = df_participation.dropna(
            subset=["id_eleccio", "territori_codi", "districte", "seccio"]
        )

    return df_votes.merge(
        df_participation[
            [
                "id_eleccio",
                "territori_codi",
                "districte",
                "seccio",
                "cens_electoral",
                "vots_valids",
                "vots_blancs",
                "vots_nuls",
                "votants",
            ]
        ],
        on=["id_eleccio", "territori_codi", "districte", "seccio"],
        how="left",
    )


def create_valid_votes_percentage_column(
    df: pd.DataFrame, remove_na: bool = True
) -> pd.DataFrame:
    """
    Create votes percentages.
    """
    return create_percentage_column(df, "vots", "vots_valids", remove_na)


def create_census_percentage_column(
    df: pd.DataFrame, remove_na: bool = True
) -> pd.DataFrame:
    """
    Create votes percentages.
    """
    return create_percentage_column(df, "vots", "cens_electoral", remove_na)


def create_total_votes_percentage_column(
    df: pd.DataFrame, remove_na: bool = True
) -> pd.DataFrame:
    """
    Create votes percentages.
    """
    return create_percentage_column(df, "vots", "votants", remove_na)


def create_percentage_column(
    df: pd.DataFrame, votes_column, divisor_column, remove_na: bool = True
) -> pd.DataFrame:
    logging.info("Creating %s_percentage.", divisor_column)

    # Make a copy of the DataFrame to ensure we're not working on a view/slice
    df_modified = df.copy()

    if remove_na:
        df_modified.dropna(subset=[divisor_column], inplace=True)
    elif df_modified[divisor_column].isna().any():
        logging.warning("'%s' contains NA values. Replacing with 0.", divisor_column)
        df_modified[divisor_column].fillna(0, inplace=True)

    df_modified[f"{divisor_column}_percentage"] = (
        df_modified[votes_column] / df_modified[divisor_column] * 100
    )

    return df_modified


def concat_dataframes(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Join dataframes.
    """
    logging.info("Joining dataframes.")
    return pd.concat(dataframes, ignore_index=True)


def divide_string_columns(
    df: pd.DataFrame, columns_to_divide: List[Dict]
) -> pd.DataFrame:
    """
    Divide columns based on regular expressions into specified new columns.

    Parameters:
    - df (pd.DataFrame): The dataframe to modify.
    - columns_to_divide (List[Dict]): A list of dictionaries specifying the column to divide,
        the new columns names, and the regex pattern.

    Returns:
    - pd.DataFrame: The modified dataframe with new columns.
    """
    logging.info("Dividing columns.")
    for column_info in columns_to_divide:
        # Ensure the column exists to avoid KeyError
        if column_info["column"] in df.columns:
            # Convert the column to string to ensure correct regex operations
            df[column_info["column"]] = df[column_info["column"]].astype(str)

            # Extract the groups based on the regex pattern
            new_data = df[column_info["column"]].str.extract(
                column_info["regex_separator"]
            )

            # Check if the number of new columns matches the number of extracted groups
            if new_data.shape[1] == len(column_info["new_columns"]):
                for i, new_col in enumerate(column_info["new_columns"]):
                    df[new_col] = new_data[i].astype(str)
            else:
                logging.warning(
                    "Extraction did not match the expected number of columns for %s",
                    column_info["column"],
                )
        else:
            logging.warning("Column %s not found in DataFrame.", column_info["column"])
    return df


def pivot_table(df: pd.DataFrame, config, prefix_columns=True) -> pd.DataFrame:
    """
    Pivot table with an option to prefix column names based on the 'columns' field.

    Args:
        df (pd.DataFrame): The DataFrame to pivot.
        config (dict): Configuration dict containing 'index', 'columns', 'values', and 'aggfunc'.
        prefix_columns (bool): If True, prefix the new column names with the name from 'columns'. Default is False.

    Returns:
        pd.DataFrame: The pivoted DataFrame with optional prefixed column names.
    """
    logging.info("Pivoting table.")
    index = config.get("index")
    columns = config.get("columns")
    values = config.get("values")
    aggfunc = config.get("aggfunc")

    # Perform the pivot operation
    df_pivoted = df.pivot_table(
        index=index, columns=columns, values=values, aggfunc=aggfunc
    )

    # Optionally prefix column names
    if prefix_columns:
        # Handling when columns are multi-level after pivot
        if isinstance(df_pivoted.columns, pd.MultiIndex):
            df_pivoted.columns = [
                "_".join([values] + list(map(str, col))).strip()
                for col in df_pivoted.columns
            ]
        else:
            df_pivoted.columns = [f"{values}_{col}" for col in df_pivoted.columns]

    return df_pivoted


def fix_total_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix total column.
    """
    logging.info("Fixing total column.")
    # Replace empty values "." with "NA"
    df["Total"] = df["Total"].replace(".", pd.NA)
    # Replace the Spanish thousand separator '.' with nothing and convert the decimal ',' to '.'
    df["Total"] = df["Total"].str.replace(".", "").str.replace(",", ".")
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce").astype("Int64")
    return df


def create_column_based_on_filename(
    df: pd.DataFrame, filename: str, column_name: str, regex: str
) -> pd.DataFrame:
    """
    Create a new column based on the filename.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - filename (str): The name of the file.
    - column_name (str): The name of the new column to create.
    - regex (str): The regular expression pattern to extract from the filename.

    Returns:
    - pd.DataFrame: The modified DataFrame with the new column added.
    """
    logging.info("Creating column based on filename.")
    df = df.copy()
    match = re.search(regex, filename)
    if match:
        df[column_name] = match.group(1)
    else:
        df[column_name] = None
        logging.warning("No match found for regex %s in filename %s.", regex, filename)
    return df


def load_data_create_column(
    filename: str,
    column_name: str,
    regex: str,
    sep: str = None,
    decimals: str = None,
    thousands: str = None,
    dtype: str = None,
) -> pd.DataFrame:
    """
    Load data and create a new column based on the filename.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - filename (str): The name of the file.
    - column_name (str): The name of the new column to create.
    - regex (str): The regular expression pattern to extract from the filename.

    Returns:
    - pd.DataFrame: The modified DataFrame with the new column added.
    """
    logging.info("Loading data and creating column based on filename.")
    df = load_data(
        filename, sep=sep, decimals=decimals, thousands=thousands, dtype=dtype
    )
    df = create_column_based_on_filename(df, filename, column_name, regex)
    return df


def aggregate_columns(df: pd.DataFrame, groups: Dict) -> pd.DataFrame:
    """
    Group columns based on a dictionary of groups adding its values.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - groups (Dict): A dictionary where keys are the new column names and values are
        lists of columns to group.

    Returns:
    - pd.DataFrame: The modified DataFrame with the new grouped columns.
    """
    logging.info("Grouping columns.")

    # Create new columns with the sum of specified columns
    for new_column, columns in groups.items():
        df[new_column] = df[columns].sum(axis=1)

    #  Drop the original columns
    columns_to_drop = [column for columns in groups.values() for column in columns]
    df.drop(columns=columns_to_drop, inplace=True)

    return df


def aggregate_rows(df: pd.DataFrame, groups_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Aggregates rows of a DataFrame based on specified grouping criteria.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be aggregated.
    - groups_info (Dict[str, Any]): A dictionary containing the following keys:
        - 'indexes' (List[str]): List of columns to group by besides the group column.
        - 'group_column' (str): The column name to be grouped.
        - 'groups' (Dict[str, List[str]]): A dictionary where keys are new group names and values
            are lists of old group names.
        - 'agg_columns' (List[str]): List of columns to be aggregated.
        - 'new_column' (str, optional): The name of the new column to be created. Defaults to the original group column name.

    Returns:
    - pd.DataFrame: The aggregated DataFrame.
    """
    # Get the parameters from the dictionary
    indexes = groups_info["indexes"]
    group_column = groups_info["group_column"]
    groups = groups_info["groups"]
    agg_columns = groups_info["agg_columns"]
    new_column = groups_info.get("new_column", group_column)

    # Create a mapping of old groups to new groups
    group_mapping = {old: new for new, olds in groups.items() for old in olds}

    # Apply the mapping to create a new grouped column
    df[new_column] = df[group_column].map(group_mapping).fillna(df[group_column])

    # Group by the new group and other specified indexes
    grouped = df.groupby(indexes + [new_column])[agg_columns].sum().reset_index()

    return grouped


def calculate_p_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the percentage of each age group within each year and mundissec,
    then removes the rows where age_groups is "Total".

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns 'year', 'mundissec', 'age_groups', and 'Total'.

    Returns:
    - pd.DataFrame: The modified DataFrame with a new column 'p_age_groups' and rows with age_groups 'Total' removed.
    """
    # Calculate the total for each year and mundissec
    total_df = df[df["age_groups"] == "Total"][["year", "mundissec", "Total"]].rename(
        columns={"Total": "total_value"}
    )

    # Merge the total values back to the original dataframe
    df = df.merge(total_df, on=["year", "mundissec"])

    # Calculate the percentage of each age group
    df["p_age_groups"] = df["Total"] / df["total_value"]

    # Remove rows where age_groups is 'Total'
    df = df[df["age_groups"] != "Total"]

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
            self.elections_data_filename = config.get("elections_data_filename")
            self.elections_days_filename = config.get("elections_days_filename")
            self.elections_participation_filename = config.get(
                "elections_participation_filename"
            )
            self.mean_income_directory = config.get("mean_income_directory")
            self.place_of_birth_directory = config.get("place_of_birth_directory")
            self.age_groups_directory = config.get("age_groups_directory")
            self.socioeconomic_index_directory = config.get(
                "socioeconomic_index_directory"
            )
            self.create_column_on_load = config.get("create_column_on_load")

            self.df = None
            if self.elections_data_filename is not None:
                self.df = load_data(self.elections_data_filename)

            self.elections_days_df = None
            if self.elections_days_filename is not None:
                self.elections_days_df = load_data(self.elections_days_filename)

            self.elections_participation_df = None
            if self.elections_participation_filename is not None:
                self.elections_participation_df = load_data(
                    self.elections_participation_filename
                )

            if self.mean_income_directory is not None:
                # List all files in the directory
                files = os.listdir(self.mean_income_directory)
                # Assuming all files in the directory are relevant and should be loaded
                self.mean_income_filenames_dfs = [
                    load_data(
                        os.path.join(self.mean_income_directory, filename),
                        decimals=",",
                        thousands=".",
                        dtype=str,
                    )
                    for filename in files
                ]

            if self.place_of_birth_directory is not None:
                # List all files in the directory
                files = os.listdir(self.place_of_birth_directory)
                # Assuming all files in the directory are relevant and should be loaded
                self.place_of_birth_filenames_dfs = [
                    load_data_create_column(
                        filename=os.path.join(self.place_of_birth_directory, filename),
                        column_name=self.create_column_on_load.get("new_column"),
                        regex=self.create_column_on_load.get("regex"),
                        sep=";",
                        decimals=",",
                        thousands=".",
                        dtype=str,
                    )
                    for filename in files
                ]

            if self.age_groups_directory is not None:
                # List all files in the directory
                files = os.listdir(self.age_groups_directory)
                # Assuming all files in the directory are relevant and should be loaded
                self.age_groups_filenames_dfs = [
                    load_data_create_column(
                        filename=os.path.join(self.age_groups_directory, filename),
                        column_name=self.create_column_on_load.get("new_column"),
                        regex=self.create_column_on_load.get("regex"),
                        sep=";",
                        decimals=",",
                        thousands=".",
                        dtype=str,
                    )
                    for filename in files
                ]

            if self.socioeconomic_index_directory is not None:
                # List all files in the directory
                files = os.listdir(self.socioeconomic_index_directory)
                # Assuming all files in the directory are relevant and should be loaded
                self.socioeconomic_index_filenames_dfs = [
                    load_data(
                        filename=os.path.join(
                            self.socioeconomic_index_directory, filename
                        ),
                        dtype=str,
                    )
                    for filename in files
                ]

            self.output_filename = config.get("output_filename")
            self.party_codes_to_fix = config.get("fix_party_codes")
            self.columns_to_drop = config.get("columns_to_drop")
            self.columns_to_rename = config.get("columns_to_rename")
            self.elections_type = config.get("elections_type")
            self.territorial_levels = config.get("territorial_levels")
            self.color_column = config.get("color_column")
            self.color_default = config.get("color_default")
            self.columns_types = config.get("columns_types")
            self.columns_null_values = config.get("columns_null_values")
            self.columns_to_divide = config.get("divide_columns")
            self.filter_by_income = config.get("filter_by_income")
            self.empty_columns_to_remove = config.get("remove_empty_rows")
            self.pivot_table = config.get("pivot_table")
            self.remove_rows_by_values = config.get("remove_rows_by_values")
            self.age_groups = config.get("group_age_groups")

            self.run_fix_party_codes = self.party_codes_to_fix is not None
            self.run_columns_null_values = self.columns_null_values is not None
            self.run_create_party_column = config.get("create_party_column")
            self.run_create_mundissec_column = config.get("create_mundissec_column")
            self.run_divide_id_eleccio = config.get("divide_id_eleccio")
            self.run_rename_columns = self.columns_to_rename is not None
            self.run_filter_by_election_type = self.elections_type is not None
            self.run_filter_by_territorial_level = self.territorial_levels is not None
            self.run_merge_election_days = self.elections_days_df is not None
            self.run_create_date_column = config.get("create_date_column")
            self.run_create_date_columns = config.get("create_date_columns")
            self.run_replace_nan_colors = self.color_column is not None
            self.run_drop_columns = self.columns_to_drop is not None
            self.run_set_column_type = self.columns_types is not None
            self.run_merge_participation_data = (
                self.elections_participation_df is not None
            )
            self.run_aggregate_duplicated_parties = config.get(
                "aggregate_duplicated_parties"
            )
            self.run_concat_mean_income_dfs = self.mean_income_directory is not None
            self.run_concat_place_of_birth_dfs = (
                self.place_of_birth_directory is not None
            )
            self.run_concat_age_groups_dfs = self.age_groups_directory is not None
            self.run_concat_socioeconomic_index_dfs = (
                self.socioeconomic_index_directory is not None
            )
            self.fix_total_column = config.get("fix_total_column")
            self.run_divide_columns = self.columns_to_divide is not None
            self.run_filter_by_income = self.filter_by_income is not None
            self.run_remove_empty_rows = self.empty_columns_to_remove is not None
            self.run_pivot_table = self.pivot_table is not None
            self.run_remove_rows_by_values = self.remove_rows_by_values is not None
            self.run_calculate_p_born_abroad = config.get("calculate_p_born_abroad")
            self.run_fix_mundissec = config.get("fix_mundissec")
            self.run_group_age_groups = self.age_groups is not None
            self.run_calculate_p_age_groups = config.get("calcutate_p_age_groups")

            data_type = config.get("data_type")
            if data_type == "elections_data":
                self.clean_elections_data()
            elif data_type == "mean_income_data":
                self.clean_mean_income_data()
            elif data_type == "place_of_birth":
                self.clean_place_of_birth()
            elif data_type == "age_groups":
                self.clean_age_groups()
            elif data_type == "socioeconomic_index":
                self.clean_socioeconomic_index()
            else:
                logging.warning("No data to clean.")

    def clean_elections_data(self):
        """
        Clean elections data.
        """
        logging.info("Cleaning elections data.")

        if self.run_fix_party_codes:
            self.df = fix_party_codes(self.df, self.party_codes_to_fix)
        if self.run_columns_null_values:
            self.df = replace_empty_rows(
                self.df, empty_columns=self.columns_null_values, value=""
            )
        if self.run_rename_columns:
            self.df = rename_columns(self.df, columns_to_rename=self.columns_to_rename)
        if self.run_filter_by_territorial_level:
            self.df = filter_by_column(
                self.df, column="id_nivell_territorial", values=self.territorial_levels
            )
        if self.run_divide_id_eleccio:
            self.df = divide_id_eleccio(self.df)
        if self.run_filter_by_election_type:
            self.df = filter_by_column(
                self.df, column="type", values=self.elections_type
            )
        if self.run_create_party_column:
            self.df = create_party_columns(self.df)
        if self.run_create_mundissec_column:
            self.df = create_mundissec_column(self.df)
        if self.run_merge_election_days:
            self.df = merge_election_days(
                self.df, df_elections_days=self.elections_days_df
            )
        if self.run_create_date_column:
            self.df = create_date_column(self.df)
        if self.run_create_date_columns:
            self.df = create_date_columns(self.df)
        if self.run_replace_nan_colors:
            self.df = replace_nan_colors(
                self.df, column=self.color_column, color=self.color_default
            )
        if self.run_drop_columns:
            self.df = drop_columns(self.df, columns_to_drop=self.columns_to_drop)
        if self.run_set_column_type:
            self.df = set_column_type(self.df, column_types=self.columns_types)

        if self.run_merge_participation_data:
            self.df = merge_participation_data(
                self.df, df_participation=self.elections_participation_df
            )
            self.df = create_valid_votes_percentage_column(self.df, remove_na=False)
            self.df = create_census_percentage_column(self.df, remove_na=False)
            self.df = create_total_votes_percentage_column(self.df, remove_na=False)
        if self.run_aggregate_duplicated_parties:
            self.df = aggregate_duplicated_parties(self.df)

        save_data(self.df, self.output_filename)

    def clean_mean_income_data(self):
        """
        Clean mean income data.
        """
        logging.info("Cleaning mean income data.")

        if self.run_concat_mean_income_dfs:
            self.df = concat_dataframes(dataframes=self.mean_income_filenames_dfs)
        if self.fix_total_column:
            self.df = fix_total_column(self.df)
        if self.run_remove_empty_rows:
            self.df = remove_empty_rows(self.df, self.empty_columns_to_remove)
        if self.run_divide_columns:
            self.df = divide_string_columns(
                self.df, columns_to_divide=self.columns_to_divide
            )
        if self.run_fix_mundissec:
            self.df = fix_mundissec(self.df)
        if self.run_filter_by_income:
            self.df = filter_by_column(
                self.df,
                column=self.filter_by_income.get("column"),
                values=self.filter_by_income.get("values"),
            )
        if self.run_rename_columns:
            self.df = rename_columns(self.df, self.columns_to_rename)
        if self.run_pivot_table:
            self.df = pivot_table(
                self.df,
                config=self.pivot_table,
            )

        save_data(self.df, self.output_filename, index=self.run_pivot_table)

    def clean_place_of_birth(self):
        """
        Clean place of birth data.
        """
        logging.info("Cleaning place of birth data.")

        if self.run_concat_place_of_birth_dfs:
            self.df = concat_dataframes(dataframes=self.place_of_birth_filenames_dfs)
        if self.run_remove_rows_by_values:
            self.df = remove_rows_by_values(self.df, self.remove_rows_by_values)
        if self.run_rename_columns:
            self.df = rename_columns(self.df, self.columns_to_rename)
        if self.run_fix_mundissec:
            self.df = fix_mundissec(self.df)
        if self.run_calculate_p_born_abroad:
            self.df = calculate_p_born_abroad(self.df)
        if self.run_pivot_table:
            self.df = pivot_table(
                self.df,
                config=self.pivot_table,
            )

        save_data(self.df, self.output_filename, index=self.run_pivot_table)

    def clean_age_groups(self):
        """
        Clean age groups data.
        """
        logging.info("Cleaning age groups data.")

        if self.run_concat_age_groups_dfs:
            self.df = concat_dataframes(dataframes=self.age_groups_filenames_dfs)
        if self.run_remove_rows_by_values:
            self.df = remove_rows_by_values(self.df, self.remove_rows_by_values)
        if self.run_rename_columns:
            self.df = rename_columns(self.df, self.columns_to_rename)
        if self.run_fix_mundissec:
            self.df = fix_mundissec(self.df)
        if self.run_group_age_groups:
            self.df = aggregate_rows(self.df, groups_info=self.age_groups)
        if self.run_calculate_p_age_groups:
            self.df = calculate_p_age_groups(self.df)
        if self.run_pivot_table:
            self.df = pivot_table(
                self.df,
                config=self.pivot_table,
            )

        save_data(self.df, self.output_filename, index=self.run_pivot_table)

    def clean_socioeconomic_index(self):
        """
        Clean socioeconomic data.
        """
        logging.info("Cleaning socioeconomic data.")

        if self.run_concat_socioeconomic_index_dfs:
            self.df = concat_dataframes(
                dataframes=self.socioeconomic_index_filenames_dfs
            )
        if self.run_remove_rows_by_values:
            self.df = remove_rows_by_values(self.df, self.remove_rows_by_values)
        if self.run_rename_columns:
            self.df = rename_columns(self.df, self.columns_to_rename)
        if self.run_pivot_table:
            self.df = pivot_table(
                self.df,
                config=self.pivot_table,
            )

        save_data(self.df, self.output_filename, index=self.run_pivot_table)
