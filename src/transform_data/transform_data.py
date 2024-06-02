"""
Transform censal sections data and results data into a single output dataframe
"""

import os
from typing import Dict, List, Tuple

import re
import logging
import pandas as pd

import geopandas as gpd
from utils.rw_files import load_data, save_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def select_important_parties(
    df: pd.DataFrame,
    n_important_parties: int = 6,
    election_column: str = "id_eleccio",
    party_code_column: str = "joined_code",
    votes_column: str = "vots",
    party_name_column: str = "joined_name",
    party_color_column: str = "joined_color",
    party_abbr_column: str = "joined_abbr",
) -> pd.DataFrame:
    """
    Select the top n parties per election and aggregate the rest as 'Other Parties'

    Args:
        df (pandas.DataFrame): DataFrame containing the election results data

    Returns:
        pandas.DataFrame: DataFrame with the top n parties per election
            and 'Other Parties' aggregated
    """
    logging.info("Selecting important parties")

    df = df.copy()
    # Determine the top n parties per election
    top_parties = (
        set()
    )  # This will hold the joined codes of top parties across all elections
    grouped = df.groupby(election_column)

    for _, group in grouped:
        # Sort parties in each election by 'votes' and select the top n
        top_in_election = (
            group.sort_values(by=votes_column, ascending=False)[[party_code_column]]
            .drop_duplicates()
            .head(n_important_parties)[party_code_column]
        )
        top_parties.update(top_in_election)

    # Identify rows that are not in the top parties
    mask = ~df[party_code_column].isin(top_parties)

    # Replace details for non-top parties to aggregate them as 'Other Parties'
    df.loc[mask, party_code_column] = 999999999
    df.loc[mask, party_name_column] = "Other Parties"
    df.loc[mask, party_color_column] = "#535353"
    df.loc[mask, party_abbr_column] = "Other"

    return df


def select_important_parties_last_election(
    df: pd.DataFrame,
    n_important_parties: int = 9,
    election_column: str = "id_eleccio",
    date_column: str = "date",  # Adding the date column as a parameter
    party_code_column: str = "joined_code",
    votes_column: str = "vots",
    party_name_column: str = "joined_name",
    party_color_column: str = "joined_color",
    party_abbr_column: str = "joined_abbr",
) -> pd.DataFrame:
    """
    Select the top n parties from the most recent election determined by the latest date
    and aggregate the rest as 'Other Parties'.

    Args:
        df (pandas.DataFrame): DataFrame containing the election results data.
        n_important_parties (int): Number of important parties to select.
        election_column (str): Column name for the election identifier.
        date_column (str): Column name for the election date.
        party_code_column (str): Column name for the party code.
        votes_column (str): Column name for the votes.
        party_name_column (str): Column name for the party name.
        party_color_column (str): Column name for the party color.
        party_abbr_column (str): Column name for the party abbreviation.

    Returns:
        pandas.DataFrame: DataFrame with the top n parties from the last election
            and 'Other Parties' aggregated.
    """
    logging.info("Selecting important parties from the most recent election")

    df = df.copy()
    # Check if necessary columns exist in DataFrame
    required_columns = [
        election_column,
        date_column,
        party_code_column,
        votes_column,
        party_name_column,
        party_color_column,
        party_abbr_column,
    ]
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"Missing required column: %s", col)
            raise ValueError(f"Missing required column: {col}")

    # Convert date column to datetime if not already
    df[date_column] = pd.to_datetime(df[date_column], errors="raise")

    # Identify the most recent election by the latest date
    latest_date = df[date_column].max()
    latest_election_df = df[df[date_column] == latest_date]

    # Sort parties by 'votes' and select the top n
    top_parties = (
        latest_election_df.sort_values(by=votes_column, ascending=False)[
            [party_code_column]
        ]
        .drop_duplicates()
        .head(n_important_parties)[party_code_column]
    )

    # Identify rows that are not in the top parties
    mask = ~df[party_code_column].isin(top_parties)

    # Replace details for non-top parties to aggregate them as 'Other Parties'
    df.loc[mask, party_code_column] = 999999999
    df.loc[mask, party_name_column] = "Other Parties"
    df.loc[mask, party_color_column] = "#535353"
    df.loc[mask, party_abbr_column] = "Other"
    return df


def select_important_parties_by_list(
    df: pd.DataFrame,
    important_parties_codes: List[int],
    party_code_column: str = "joined_code",
    party_name_column: str = "joined_name",
    party_color_column: str = "joined_color",
    party_abbr_column: str = "joined_abbr",
) -> pd.DataFrame:
    # Identify rows that are not in the top parties
    mask = ~df[party_code_column].isin(important_parties_codes)

    # Replace details for non-top parties to aggregate them as 'Other Parties'
    df.loc[mask, party_code_column] = 999999999
    df.loc[mask, party_name_column] = "Other Parties"
    df.loc[mask, party_color_column] = "#535353"
    df.loc[mask, party_abbr_column] = "Other"
    return df


def add_missing_party_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Adds missing party columns to the GeoDataFrame.
    Columns are identified by the pattern `VARIABLENAME_PARTYCODE_ELECTIONSID`.
    Missing columns are added with their values filled as `0.0`.

    :param gdf: The original GeoDataFrame with existing columns.
    :return: The GeoDataFrame with all possible party columns present.
    """
    logging.info("Add missing party columns")

    # Function to parse columns into three components
    def parse_column_name(col_name):
        parts = col_name.split("_")
        election_id = parts[-1]
        party_code = parts[-2]
        variable_name = "_".join(parts[:-2])  # Handle underscores within VARIABLENAME
        return variable_name, party_code, election_id

    # Extract unique patterns
    existing_patterns = {parse_column_name(col) for col in gdf.columns if "_" in col}

    # Create sets for unique components
    unique_vars = {x[0] for x in existing_patterns}
    unique_parties = {x[1] for x in existing_patterns}
    unique_elections = {x[2] for x in existing_patterns}

    # Generate all combinations
    all_combinations = {
        f"{var}_{party}_{election}"
        for var in unique_vars
        for party in unique_parties
        for election in unique_elections
    }

    # Determine which combinations are missing
    existing_columns = set(gdf.columns)
    missing_columns = all_combinations - existing_columns

    # Create a DataFrame to hold the missing columns
    missing_df = pd.DataFrame(0.0, index=gdf.index, columns=sorted(missing_columns))

    # Concatenate the new columns to the original GeoDataFrame
    gdf = pd.concat([gdf, missing_df], axis=1)

    return gdf


def manually_group_parties(
    df: pd.DataFrame,
    parties_to_group: Dict[int, List[int]],
    column_name: str = "joined_code",
) -> pd.DataFrame:
    """
    Manually group parties in the DataFrame

    Args:
        df (pandas.DataFrame): DataFrame containing the election results data
        parties_to_group (Dict): Dictionary containing the parties to group.

    Returns:
        pandas.DataFrame: DataFrame with the parties grouped as specified in the input dictionary
    """
    logging.info("Manually grouping parties")

    for group_name, group_parties in parties_to_group.items():
        # Get the joined codes of the parties to group
        joined_codes = df[df[column_name].isin(group_parties)][column_name].unique()

        # Replace the joined code of the parties to group with the joined code of the group
        df.loc[df[column_name].isin(joined_codes), column_name] = group_name

    return df


def create_timeseries_df(df, columns_pattern_to_long: str = None):
    # Set "MUNDISSEC" as index
    df = df.set_index("MUNDISSEC")

    if columns_pattern_to_long is not None:
        columns_to_long = [col for col in df.columns if columns_pattern_to_long in col]
        df_long = df[columns_to_long]
        df_fixed = df.loc[:, ~df.columns.isin(columns_to_long)]

    # Melt the DataFrame to long format
    # Creates a new column named 'variable' that contains the column names
    # For each census section (MUNDISSEC), there will be multiple rows, one for each variable
    df_long = df_long.reset_index().melt(id_vars=["MUNDISSEC"])
    # Extract variable name and electionid from 'variable' column
    df_long["electionid"] = df_long["variable"].apply(lambda x: x.split("_")[-1])
    df_long["variable_name"] = df_long["variable"].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )

    # Drop the original 'variable' column as it's now redundant
    df_long.drop(columns="variable", inplace=True)

    # Set Index using MUNDISSEC and electionid
    df_long["electionid_mundissec"] = df_long["electionid"] + "_" + df_long["MUNDISSEC"]
    df_long = df_long.set_index("electionid_mundissec")

    # Pivot the table to wide format
    df_long.drop(columns=["electionid", "MUNDISSEC"], inplace=True)
    df_timeseries = df_long.pivot(columns="variable_name")

    # Flatten the MultiLevel column index
    # Ensure that the columns are on the same level by joining level names
    df_timeseries.columns = [col[1] for col in df_timeseries.columns.values]

    df_timeseries["mundissec"] = df_timeseries.index.str.split("_").str[1]
    df_timeseries = df_timeseries.merge(
        df_fixed,
        left_on="mundissec",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    df_timeseries.drop(columns="mundissec", inplace=True)

    return df_timeseries


def extract_year(index_string: str) -> int:
    """
    Extracts the year from the given index string.

    Args:
        index_string (str): The index string containing the year.

    Returns:
        int: The extracted year as an integer.
    """
    # Assuming the year is four digits and starts right after the first character
    digits_start = 1
    # The year will then be the first four digits after this point
    return int(index_string[digits_start : digits_start + 4])


def get_max_columns_df(
    original_df: pd.DataFrame, variables_df: pd.DataFrame, year_info: str = "index"
) -> Tuple[pd.DataFrame, int]:
    """
    Returns a DataFrame with only columns that contain the maximum year and the maximum year value.

    Parameters:
    original_df (DataFrame): The DataFrame containing the original data.
    variables_df (DataFrame): The DataFrame containing the variables data.
    year_info (str): Where the information about the year is stored.
        It can be either "index" or "column".

    Returns:
    Tuple[DataFrame, int]: A tuple containing the resulting DataFrame with only columns that contain the max year
    and the maximum year value.
    """
    if year_info == "index":
        # Extract the year from the index
        max_year_original = max(original_df.index.map(extract_year))
    elif year_info == "column":
        # Extract the year from the column names
        max_year_original = max(
            [
                int(match.group())
                for col in original_df.columns
                if (match := re.search(r"\d{4}", col)) is not None
            ]
        )
    else:
        raise ValueError("year_info must be either 'index' or 'column'")

    max_year_variables = max(
        [
            int(re.search(r"\d+", col).group())
            for col in variables_df.columns
            if re.search(r"\d+", col) is not None
        ]
    )
    max_year = min(max_year_original, max_year_variables)
    max_year_columns = [col for col in variables_df.columns if str(max_year) in col]

    # Resulting DataFrame with only columns that contain the max year
    return variables_df[max_year_columns], max_year


def rename_last_born_abroad_df(df: pd.DataFrame, max_year: int) -> pd.DataFrame:
    """
    Renames the column 'p_born_abroad_{max_year}' to '{max_year}_born_abroad' in the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        max_year (int): The maximum year.

    Returns:
        pd.DataFrame: The DataFrame with the renamed column.
    """
    df_renamed = df.rename(
        columns={f"p_born_abroad_{max_year}": f"{max_year}_born_abroad"},
        inplace=False,
    )

    return df_renamed


def rename_age_groups_columns(df: pd.DataFrame, max_year: int) -> pd.DataFrame:
    """
    Renames the age group columns in the given DataFrame based on the maximum year.

    Args:
        df (pd.DataFrame): The DataFrame containing the age group columns.
        max_year (int): The maximum year to be included in the column names.

    Returns:
        pd.DataFrame: The DataFrame with renamed age group columns.
    """
    # Find all columns that match the pattern "p_age_groups_{age_group}_{max_year}"
    age_group_columns = [
        col
        for col in df.columns
        if col.startswith("p_age_groups_") and col.endswith(f"_{max_year}")
    ]
    # Create a mapping from the old column names to the new column names
    rename_mapping: Dict[str, str] = {
        col: f"{max_year}_{col.split('_')[-2]}" for col in age_group_columns
    }
    # Rename the columns to {max_year}_{age_group}
    df_renamed = df.rename(columns=rename_mapping, inplace=False)
    return df_renamed


def rename_mean_income_columns(df: pd.DataFrame, max_year: int) -> pd.DataFrame:
    """
    Renames the columns of a DataFrame to "{year}_mean_income".

    Args:
        df (pd.DataFrame): The DataFrame to rename the columns of.
        max_year (int): The maximum year for which the mean income column is available.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """
    df_renamed = df.rename(
        columns={f"mean_income_{max_year}": f"{max_year}_mean_income"},
        inplace=False,
    )
    return df_renamed


def rename_ist_column(df: pd.DataFrame, max_year: int) -> pd.DataFrame:
    """
    Renames the column "valor_Índex socioeconòmic territorial_{max_year}" to "{max_year}_ist" in the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        max_year (int): The maximum year value.

    Returns:
        pd.DataFrame: The modified DataFrame with the renamed column.
    """
    df_renamed = df.rename(
        columns={
            f"valor_Índex socioeconòmic territorial_{max_year}": f"{max_year}_ist"
        },
        inplace=False,
    )
    return df_renamed


def ensure_joined_abbr_consistency(df):
    # Group by 'joined_code'
    grouped = df.groupby("joined_code")

    # Iterate through each group
    for name, group in grouped:
        if name == 999999999:
            consistent_clean_abbr = "OTH"
            consistent_uppercased_abbr = "OTH"
            consistent_name = "Other Parties"
            consistent_abbr = "Other"
        else:
            consistent_clean_abbr = group["joined_clean_abbr"].iloc[-1]
            consistent_uppercased_abbr = consistent_clean_abbr.upper()
            consistent_name = group["joined_name"].iloc[-1]
            consistent_abbr = group["joined_abbr"].iloc[-1]

        # Ensure consistency by keeping the first unique 'joined_clean_abbr' value
        df.loc[df["joined_code"] == name, "joined_clean_abbr"] = (
            consistent_uppercased_abbr
        )
        df.loc[df["joined_code"] == name, "joined_name"] = consistent_name
        df.loc[df["joined_code"] == name, "joined_abbr"] = consistent_abbr

    return df


def shorten_column_name(column_name: str) -> str:
    """
    Shorten a column name according to the specified strategy:
    1. Remove the word "percentage_"
    2. Cap uppercase words surrounded by underscores to 2 characters
    3. Replace lowercase words separated by underscores with their initials
    4. Remove underscores

    Args:
        column_name (str): The original column name.

    Returns:
        str: The shortened column name.
    """
    # Step 1: Remove the word "percentage_"
    column_name = column_name.replace("percentage_", "")

    # Separate the suffix (anything after the last underscore)
    parts = column_name.rsplit("_", 1)
    main_part = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""

    # Step 2: Cap uppercase words surrounded by underscores to 2 characters
    parts = main_part.split("_")
    new_parts = []
    for part in parts:
        if part.isupper():
            new_parts.append(part[:2])
        elif part.islower():
            # Step 3: Replace lowercase words separated by an underscore with initials
            initials = "".join([word[0] for word in part.split("_")])
            new_parts.append(initials)
        else:
            new_parts.append(part)

    # Step 4: Remove underscores
    shortened_name = "".join(new_parts)

    # Combine with suffix and ensure the name is within the 10 character limit
    final_name = shortened_name + suffix
    return final_name[:10]


def shorten_column_names(columns: List[str]) -> Dict[str, str]:
    """
    Apply the column name shortening strategy to a list of column names.

    Args:
        columns (List[str]): The original list of column names.

    Returns:
        Dict[str, str]: A dictionary mapping original column names to shortened names.
    """
    return {col: shorten_column_name(col) for col in columns if len(col) > 10}


class TransformData:
    """
    Transform censal sections data and results data into a single output dataframe
    """

    def __init__(
        self,
        censal_sections_path: str,
        results_path: str,
        adjacency_matrix_path: str,
        output_path: str = "../data/output/censal_sections_results",
        born_abroad_path: str = "../data/processed/place_of_birth_clean_data.pkl",
        age_groups_path: str = "../data/processed/age_groups_clean_data.pkl",
        mean_income_path: str = "../data/processed/mean_income_clean_data.pkl",
        socioeconomic_index_path: str = "../data/processed/socioeconomic_index_clean_data.pkl",
        start_year: int = 1975,
        end_year: int = 2024,
        # n_important_parties: int = 9,
        transform_to_timeseries: bool = False,
        add_election_type: int = 0,  # 0: no election type, 1: add election type as one column, 2: add election type as one column per election type
        add_born_abroad: bool = False,
        add_mean_income: bool = False,
        add_age_groups: bool = False,
        add_socioeconomic_index: bool = False,
    ) -> None:
        """
        Initialize the class with the paths to the censal sections and results data

        Args:
            censal_sections_path (str): Path to the censal sections data file
            results_path (str): Path to the election results data file
            output_path (str, optional): Path to save the output dataframe.
                Defaults to "../data/output/censal_sections_results".
        """
        self.censal_sections_gdf = load_data(censal_sections_path)
        self.results_df = load_data(results_path)
        self.adjacency_matrix = load_data(adjacency_matrix_path)
        self.output_path = output_path
        self.start_year = start_year
        self.end_year = end_year
        # self.n_important_parties = n_important_parties
        self.transform_to_timeseries = transform_to_timeseries
        self.add_election_type = add_election_type
        self.add_born_abroad = add_born_abroad
        self.add_mean_income = add_mean_income
        self.add_age_groups = add_age_groups
        self.add_socioeconomic_index = add_socioeconomic_index

        if self.add_born_abroad:
            self.born_abroad_df = load_data(born_abroad_path)
        if self.add_age_groups:
            self.age_groups_df = load_data(age_groups_path)
        if self.add_mean_income:
            self.mean_income_df = load_data(mean_income_path)
        if self.add_socioeconomic_index:
            self.socioeconomic_index_df = load_data(socioeconomic_index_path)

    def transform_data(self) -> None:
        """
        Transform censal sections data and results data into a single output dataframe
        """

        # filter the results_df to only include the years between start_year and end_year
        self.results_df = self.results_df[
            self.results_df["year"].astype(int).between(self.start_year, self.end_year)
        ]

        self.results_df = manually_group_parties(
            self.results_df,
            {
                6: [264],  # PSC
                10: [698, 610],  # ERC
                1003: [649, 1030],  # CUP, Front Republicà
                1013: [
                    331,
                    1008,
                    1001,
                    1015,
                    1096,
                    640,
                    709,
                    1099,
                    739,
                ],  # Comuns, Podemos, EUiA, ICV
                10831: [
                    12,
                    373,
                    1000,
                    10841,
                    1007,
                    412,
                    1097,
                ],  # Junts, JxSí, CiU, CDC, PDeCAT
            },
        )

        important_parties = [
            6,  # PSC
            10,  # ERC
            86,  # PP
            301,  # Cs
            693,  # VOX
            1003,  # CUP, Front Republicà
            1013,  # Comuns, Podemos, EUiA, ICV
            10831,  # Junts, JxSí, CiU, CDC, PDeCAT
        ]
        important_parties_df = select_important_parties_by_list(
            self.results_df, important_parties
        )

        important_parties_df = ensure_joined_abbr_consistency(important_parties_df)

        # Pivot the DataFrame
        logging.info("Pivoting the DataFrame")
        results_wide_df = important_parties_df.pivot_table(
            index="mundissec",
            columns=["joined_clean_abbr", "id_eleccio"],
            values=[
                # "vots",
                "votants_percentage",
                "vots_valids_percentage",
                "cens_electoral_percentage",
            ],
        )

        # Flattening the MultiIndex Columns
        results_wide_df.columns = [
            "_".join(map(str, col)).strip() for col in results_wide_df.columns.values
        ]

        # Merge censal sections and results data
        merged_gdf = self.censal_sections_gdf.merge(
            results_wide_df, left_on="MUNDISSEC", right_on="mundissec", how="inner"
        )

        logging.info("Merging done")

        # Convert all columns of type 'Float64Dtype' to 'float64' to avoid problems with GeoPandas
        for col in merged_gdf.select_dtypes(include=["Float64"]).columns:
            merged_gdf[col] = merged_gdf[col].fillna(0.0).astype("float64")

        logging.info("Converting columns to float64 done")

        merged_gdf = add_missing_party_columns(merged_gdf)

        merged_gdf = self.compute_adjacencies_features(
            merged_gdf, "cens_electoral_percentage"
        )

        logging.info("Computing adjacencies done")

        if self.add_born_abroad:
            # Resulting DataFrame with only columns that contain the max year
            last_born_abroad_df, max_year = get_max_columns_df(
                merged_gdf, self.born_abroad_df, year_info="column"
            )
            # Rename column "p_born_abroad_{max_year}" to "{max_year}_born_abroad"
            last_born_abroad_df = rename_last_born_abroad_df(
                last_born_abroad_df, max_year
            )
            merged_gdf = merged_gdf.merge(
                last_born_abroad_df, left_on="MUNDISSEC", right_index=True, how="left"
            )

        if self.add_age_groups:
            # Resulting DataFrame with only columns that contain the max year
            last_age_groups_df, max_year = get_max_columns_df(
                merged_gdf, self.age_groups_df, year_info="column"
            )
            # Rename the columns that match the pattern "p_age_groups_{age_group}_{max_year}"
            # to {max_year}_{age_group}
            last_age_groups_df = rename_age_groups_columns(
                last_age_groups_df, max_year=max_year
            )
            merged_gdf = merged_gdf.merge(
                last_age_groups_df, left_on="MUNDISSEC", right_index=True, how="left"
            )

        if self.add_mean_income:
            # Resulting DataFrame with only columns that contain the max year
            last_mean_income_df, max_year = get_max_columns_df(
                merged_gdf, self.mean_income_df, year_info="column"
            )
            # Rename column "mean_income_{max_year}" to "{max_year}_mean_income"
            last_mean_income_df = rename_mean_income_columns(
                last_mean_income_df, max_year
            )
            merged_gdf = merged_gdf.merge(
                last_mean_income_df, left_on="MUNDISSEC", right_index=True, how="left"
            )

        if self.add_socioeconomic_index:
            # Resulting DataFrame with only columns that contain the max year
            last_socioeconomic_index_df, max_year = get_max_columns_df(
                merged_gdf, self.socioeconomic_index_df, year_info="column"
            )
            # Rename column "socioeconomic_index_{max_year}" to "{max_year}_ist"
            last_socioeconomic_index_df = rename_ist_column(
                last_socioeconomic_index_df, max_year
            )
            merged_gdf = merged_gdf.merge(
                last_socioeconomic_index_df,
                left_on="MUNDISSEC",
                right_index=True,
                how="left",
            )

        if self.transform_to_timeseries:
            # Drop columns that are not necessary for the timeseries
            filtered_df = merged_gdf.drop(
                columns=["geometry", "MUNICIPI", "DISTRICTE", "SECCIO"]
            )
            # Drop `votants_percentage_*` and `vots_valids_percentage_*` sets of columns
            filtered_df = filtered_df.loc[
                :, ~filtered_df.columns.str.contains("votants_percentage_")
            ]
            filtered_df = filtered_df.loc[
                :, ~filtered_df.columns.str.contains("vots_valids_percentage_")
            ]
            timeseries_df = create_timeseries_df(
                filtered_df, columns_pattern_to_long="cens_electoral_percentage"
            )

            if self.add_election_type != 0:
                # Parse the index to extract mundissec and electionid
                timeseries_index_df = (
                    timeseries_df.index.to_series()
                    .str.split("_", expand=True)
                    .rename(columns={0: "id_eleccio", 1: "mundissec"})
                )

                # Merge on 'mundissec' and 'electionid' to find corresponding 'type'
                merged_df = pd.merge(
                    timeseries_index_df[["id_eleccio"]],
                    self.results_df[["id_eleccio", "type"]].drop_duplicates(),
                    on=["id_eleccio"],
                    how="left",
                )
                merged_df.set_index(timeseries_index_df.index, inplace=True)

                if self.add_election_type == 1:
                    # Convert type to categorical and then to codes
                    merged_df["election_type"] = pd.Categorical(merged_df["type"])
                    merged_df["election_type"] = merged_df["election_type"].cat.codes
                    timeseries_df = timeseries_df.join(merged_df[["election_type"]])

                elif self.add_election_type == 2:
                    # Create one-hot encoding for type
                    type_dummies = pd.get_dummies(
                        merged_df["type"], prefix="election_type", dtype=int
                    )
                    # Join the one-hot encoded columns back to the timeseries_df
                    timeseries_df = timeseries_df.join(type_dummies)

            save_data(
                timeseries_df,
                f"../data/output/timeseries_{self.start_year}_{self.end_year}_{self.add_election_type}_{self.add_born_abroad}_{self.add_age_groups}_{self.add_mean_income}_{self.add_socioeconomic_index}",
                index=True,
            )

        # ## Save shapefile: TODO: move to utils.rw_files
        # # Apply the function to all columns in the GeoDataFrame
        # column_mapping = shorten_column_names(merged_gdf.columns)
        # # Rename the columns in the GeoDataFrame
        # merged_gdf = merged_gdf.rename(columns=column_mapping)
        # os.makedirs("../data/output/spatial", exist_ok=True)
        # merged_gdf.to_file(
        #     f"../data/output/spatial/spatial_{self.start_year}_{self.end_year}_{self.add_election_type}_{self.add_born_abroad}_{self.add_age_groups}_{self.add_mean_income}_{self.add_socioeconomic_index}.shp"
        # )

        logging.info("Data transformation done")

    def get_adjacent_sections(self, census_section: str) -> List[str]:
        """
        Given a census section (MUNDISSEC code), return the list of adjacent census sections.

        Args:
            census_section (str): The MUNDISSEC code of the census section.

        Returns:
            List[str]: A list of MUNDISSEC codes for adjacent census sections.
        """
        if census_section not in self.adjacency_matrix.index:
            raise ValueError(
                f"Section {census_section} not found in the adjacency matrix."
            )

        adjacent_sections = self.adjacency_matrix.loc[census_section]
        return adjacent_sections[adjacent_sections == 1].index.tolist()

    def calculate_adjacencies_values(
        self, df: pd.DataFrame, census_section: str, column_pattern: str
    ) -> pd.Series:
        """
        Calculate mean values of specified columns for adjacent census sections.

        Args:
            census_section (str): The MUNDISSEC code of the census section to analyze.
            column_pattern (str): The pattern to match columns for which to compute mean values.

        Returns:
            pd.Series: A Series with the mean values of specified columns for adjacent sections.
        """
        # Obtain the set of columns that match column_pattern
        matching_columns = [col for col in df.columns if column_pattern in col]

        # Get the list of adjacent sections
        adjacent_sections = self.get_adjacent_sections(census_section)

        # Compute df with columns
        electoral_df = df[matching_columns]

        # Get a subset of df with the values to compute
        values_df = electoral_df[electoral_df.index.isin(adjacent_sections)]

        # For each column, compute the mean
        result = values_df.mean()

        return result

    def compute_adjacencies_features(
        self, df: pd.DataFrame, column_pattern: str
    ) -> pd.DataFrame:
        """
        Compute new adjacencies features for the DataFrame.

        Args:
            df (pd.DataFrame): The original DataFrame.
            column_pattern (str): The pattern to match columns for which to compute mean values.

        Returns:
            pd.DataFrame: A DataFrame with the original data and new adjacencies features.
        """
        logging.info("Computing adjacency features")

        # Make a copy of the original DataFrame
        df_copy = df.copy()
        df_copy.set_index("MUNDISSEC", inplace=True)

        # Initialize a DataFrame to hold the new adjacency features
        new_features = pd.DataFrame(index=df_copy.index)

        # Get the list of columns that match the pattern
        matching_columns = [col for col in df.columns if column_pattern in col]
        new_columns = ["adj_" + col for col in matching_columns]

        # Iterate over each section and compute adjacency values
        for census_section in df_copy.index:
            adj_values = self.calculate_adjacencies_values(
                df_copy, census_section, column_pattern
            )
            adj_values.index = new_columns
            new_features.loc[census_section, new_columns] = adj_values

        # Merge the new features with the original DataFrame copy
        df_copy = df_copy.join(new_features)
        df_copy.set_index(df.index, inplace=True)
        df_copy["MUNDISSEC"] = df["MUNDISSEC"]

        return df_copy
