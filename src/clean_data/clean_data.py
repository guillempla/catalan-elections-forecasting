"""
Clean data and save it as a CSV file.
"""

import os
import re
from typing import List, Dict

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
    # Identifying duplicates
    duplicates_mask = df.duplicated(
        subset=["mundissec", "party_code", "nom_eleccio"], keep=False
    )
    duplicated_party_codes = df[duplicates_mask]

    # Define how each column should be aggregated
    aggregations = {
        "vots": "sum",
        "escons": "sum",
        "vots_valids": "sum",
        "vots_blancs": "sum",
        "vots_nuls": "sum",
        "votants": "sum",
        "vots_valids_percentage": "mean",  # TODO: recalculate these percentages after aggregation
        "cens_electoral_percentage": "mean",  # TODO: recalculate these percentages after aggregation
        "votants_percentage": "mean",  # TODO: recalculate these percentages after aggregation
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
        duplicated_party_codes.groupby(["mundissec", "party_code", "nom_eleccio"])
        .agg(aggregations)
        .reset_index()
    )

    # Merge the aggregated results back to the original dataframe
    # First, drop duplicates from the original df
    df_cleaned = df.drop(df[duplicates_mask].index)

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
    filter_rows = (df["id_nivell_territorial"].isin(["DM", "ME", "MU", "SE"])) & (
        df["territori_codi"] == "08187"
    )

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


def pivot_table(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Pivot table.
    """
    logging.info("Pivoting table.")
    index = config.get("index")
    columns = config.get("columns")
    values = config.get("values")
    aggfunc = config.get("aggfunc")

    df = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
    return df


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
    df = load_data(filename, decimals=decimals, thousands=thousands, dtype=dtype)
    df = create_column_based_on_filename(df, filename, column_name, regex)
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
                        column_name=self.create_column_on_load.get("year"),
                        regex=self.create_column_on_load.get("regex"),
                        decimals=",",
                        thousands=".",
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
            self.run_concat_mean_income_dfs = self.mean_income_filenames_dfs is not None
            self.run_concat_place_of_birth_dfs = (
                self.place_of_birth_filenames_dfs is not None
            )
            self.fix_total_column = config.get("fix_total_column")
            self.run_divide_columns = self.columns_to_divide is not None
            self.run_filter_by_income = self.filter_by_income is not None
            self.run_remove_empty_rows = self.empty_columns_to_remove is not None
            self.run_pivot_table = self.pivot_table is not None

            data_type = config.get("data_type")
            if data_type == "elections_data":
                self.clean_elections_data()
            elif data_type == "mean_income_data":
                self.clean_mean_income_data()
            elif data_type == "place_of_birth":
                self.clean_place_of_birth()
            elif data_type == "socioeconomic_data":
                self.clean_socioeconomic_data()
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
            self.df = create_valid_votes_percentage_column(self.df)
            self.df = create_census_percentage_column(self.df)
            self.df = create_total_votes_percentage_column(self.df)
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
        if self.run_filter_by_income:
            self.df = filter_by_column(
                self.df,
                column=self.filter_by_income.get("column"),
                values=self.filter_by_income.get("values"),
            )
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

        # delete rows "País de nacimiento" == "TOTAL" and "Sexo" == "Hombres" and "Sexo" == "Mujeres"

        # change column name "Seccion" to "mundissec"

        # Add calculated column "p_bonr_abroad" = "Nacidos en el extranjero" / "Total Población"

        # Remove columns "Sexo", "País de nacimiento" and "Total"

        # Pivot table to have "p_born_abroad" by "year"

        save_data(self.df, self.output_filename, index=self.run_pivot_table)

    def clean_socioeconomic_data(self):
        """
        Clean socioeconomic data.
        """
        logging.info("Cleaning socioeconomic data.")

        pass
