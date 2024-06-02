"""
Transform censal sections data and results data into a single output dataframe
"""

import os
from typing import Dict, List, Tuple

import re
import logging
import numpy as np
import pandas as pd

# import numpy as np
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


def create_timeseries_df(df):
    # Drop `df` columns `vots_*`, `votants_percentage_*` and `vots_valids_percentage_*`
    df_filtered = df.loc[:, ~df.columns.str.contains("vots_")]
    df_filtered = df_filtered.loc[
        :, ~df_filtered.columns.str.contains("votants_percentage_")
    ]
    df_filtered = df_filtered.loc[
        :, ~df_filtered.columns.str.contains("vots_valids_percentage_")
    ]

    # Set "MUNDISSEC" as index
    df_filtered = df_filtered.set_index("MUNDISSEC")

    # Remove census section identifier columns
    df_filtered = df_filtered.drop(columns=["MUNICIPI", "DISTRICTE", "SECCIO"])

    #  Melt the DataFrame to long format
    df_long = df_filtered.reset_index().melt(id_vars=["MUNDISSEC"])

    # Extract variable name and electionid from 'variable' column
    df_long["electionid"] = df_long["variable"].apply(lambda x: x.split("_")[-1])
    df_long["variable_name"] = df_long["variable"].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )

    # Drop the original 'variable' column as it's now redundant
    df_long.drop(columns="variable", inplace=True)

    # Set Index using MUNDISSEC and electionid
    df_long["electionid_mundissec"] = df_long["electionid"] + "_" + df_long["MUNDISSEC"]
    df_long = df_long.drop(columns=["electionid", "MUNDISSEC"])
    df_long = df_long.set_index("electionid_mundissec")

    # Pivot the table to wide format (if necessary, depending on how you want to view/use the data)
    df_timeseries = df_long.pivot(columns="variable_name")

    # Flatten the MultiLevel column index
    # Ensure that the columns are on the same level by joining level names
    df_timeseries.columns = [col[1] for col in df_timeseries.columns.values]

    return df_timeseries


def extract_year(index_string):
    # Assuming the year is four digits and starts right after the first character
    digits_start = 1
    # The year will then be the first four digits after this point
    return int(index_string[digits_start : digits_start + 4])


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
        output_path: str = "../data/output/censal_sections_results",
        born_abroad_path: str = "../data/processed/place_of_birth_clean_data.pkl",
        age_groups_path: str = "../data/processed/age_groups_clean_data.pkl",
        mean_income_path: str = "../data/processed/mean_income_clean_data.pkl",
        socioeconomic_index_path: str = "../data/processed/socioeconomic_index_clean_data.pkl",
        start_year: int = 1975,
        end_year: int = 2024,
        n_important_parties: int = 9,
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
        self.output_path = output_path
        self.start_year = start_year
        self.end_year = end_year
        self.n_important_parties = n_important_parties
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

        # important_parties_df = select_important_parties(
        #     self.results_df,
        #     n_important_parties=self.n_important_parties,
        #     party_abbr_column="joined_clean_abbr",
        # )
        # important_parties = select_important_parties_last_election(
        #     self.results_df, n_important_parties=self.n_important_parties
        # )
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

        # Replace 'Infinity' values with a suitable finite number (e.g., 0)
        results_wide_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        logging.info("Replacing 'Infinity' values done")

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

        if self.transform_to_timeseries:
            timeseries_df = merged_gdf.drop(columns=["geometry"])
            timeseries_df = create_timeseries_df(timeseries_df)

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

            if self.add_born_abroad:
                max_year = max(timeseries_df.index.map(extract_year))
                max_year_columns = [
                    col for col in self.born_abroad_df.columns if str(max_year) in col
                ]

                # Resulting DataFrame with only columns that contain the max year
                last_born_abroad_df = self.born_abroad_df[max_year_columns]

                # Rename column "p_born_abroad_{max_year}" to "{max_year}_p_born_abroad"
                last_born_abroad_df.rename(
                    columns={f"p_born_abroad_{max_year}": f"{max_year}_p_born_abroad"},
                    inplace=True,
                )

                # Extract 'mundissec' from the index of timeseries_df
                # This lambda function splits the index string at the underscore and takes the second part,
                # which is 'mundissec'
                timeseries_df["mundissec"] = timeseries_df.index.to_series().apply(
                    lambda x: x.split("_")[1]
                )

                # Ensure data types are compatible
                timeseries_df["mundissec"] = timeseries_df["mundissec"].astype(str)

                # If last_born_abroad_df uses 'mundissec' as its index,
                # reset this index and ensure it's named correctly
                if (
                    last_born_abroad_df.index.name == "mundissec"
                    or "mundissec" in last_born_abroad_df.columns
                ):
                    last_born_abroad_df.reset_index(inplace=True)
                    if last_born_abroad_df.index.name == "mundissec":
                        last_born_abroad_df.rename(
                            columns={last_born_abroad_df.index.name: "mundissec"},
                            inplace=True,
                        )
                last_born_abroad_df["mundissec"] = last_born_abroad_df[
                    "mundissec"
                ].astype(
                    str
                )  # Ensure data type matches

                # Merge on 'mundissec' while keeping timeseries_df's original index
                aux_timeseries_df = timeseries_df.copy()
                aux_timeseries_df.reset_index(inplace=True, drop=True)
                merged_df = pd.merge(
                    aux_timeseries_df,
                    last_born_abroad_df,
                    on="mundissec",
                    how="left",
                )

                # Ensure that timeseries_df index is carried over
                merged_df.set_index(timeseries_df.index, inplace=True)
                timeseries_df = merged_df

            if self.add_age_groups:
                max_year = max(timeseries_df.index.map(extract_year))
                max_year_columns = [
                    col for col in self.age_groups_df.columns if str(max_year) in col
                ]

                # Resulting DataFrame with only columns that contain the max year
                last_age_groups_df = self.age_groups_df[max_year_columns]

                # Find all columns that match the pattern "p_age_groups_{age_group}_{max_year}"
                age_group_columns = [
                    col
                    for col in last_age_groups_df.columns
                    if col.startswith("p_age_groups_") and col.endswith(f"_{max_year}")
                ]
                # Create a mapping from the old column names to the new column names
                rename_mapping = {
                    col: f"{max_year}_p_{col.split('_')[-2]}"
                    for col in age_group_columns
                }
                # Rename the columns to {max_year}_p_{age_group}
                last_age_groups_df.rename(columns=rename_mapping, inplace=True)

                # Extract 'mundissec' from the index of timeseries_df
                # This lambda function splits the index string at the underscore and takes the second part,
                # which is 'mundissec'
                timeseries_df["mundissec"] = timeseries_df.index.to_series().apply(
                    lambda x: x.split("_")[1]
                )

                # Ensure data types are compatible
                timeseries_df["mundissec"] = timeseries_df["mundissec"].astype(str)

                # If last_age_groups_df uses 'mundissec' as its index,
                # reset this index and ensure it's named correctly
                if (
                    last_age_groups_df.index.name == "mundissec"
                    or "mundissec" in last_age_groups_df.columns
                ):
                    last_age_groups_df.reset_index(inplace=True)
                    if last_age_groups_df.index.name == "mundissec":
                        last_age_groups_df.rename(
                            columns={last_age_groups_df.index.name: "mundissec"},
                            inplace=True,
                        )
                last_age_groups_df["mundissec"] = last_age_groups_df[
                    "mundissec"
                ].astype(
                    str
                )  # Ensure data type matches

                # Merge on 'mundissec' while keeping timeseries_df's original index
                aux_timeseries_df = timeseries_df.copy()
                aux_timeseries_df.reset_index(inplace=True, drop=True)
                merged_df = pd.merge(
                    aux_timeseries_df,
                    last_age_groups_df,
                    on="mundissec",
                    how="left",
                )

                # Ensure that timeseries_df index is carried over
                merged_df.set_index(timeseries_df.index, inplace=True)
                timeseries_df = merged_df

            if self.add_mean_income:
                max_year = max(timeseries_df.index.map(extract_year))
                max_year_columns = [
                    col for col in self.mean_income_df.columns if str(max_year) in col
                ]

                # Resulting DataFrame with only columns that contain the max year
                last_mean_income_df = self.mean_income_df[max_year_columns]

                # Rename column "mean_income_{max_year}" to "{max_year}_mean_income"
                last_mean_income_df.rename(
                    columns={f"mean_income_{max_year}": f"{max_year}_mean_income"},
                    inplace=True,
                )

                # Extract 'mundissec' from the index of timeseries_df
                # This lambda function splits the index string at the underscore and takes the second part,
                # which is 'mundissec'
                timeseries_df["mundissec"] = timeseries_df.index.to_series().apply(
                    lambda x: x.split("_")[1]
                )

                # Ensure data types are compatible
                timeseries_df["mundissec"] = timeseries_df["mundissec"].astype(str)

                # If last_mean_income_df uses 'mundissec' as its index,
                # reset this index and ensure it's named correctly
                if (
                    last_mean_income_df.index.name == "mundissec"
                    or "mundissec" in last_mean_income_df.columns
                ):
                    last_mean_income_df.reset_index(inplace=True)
                    if last_mean_income_df.index.name == "mundissec":
                        last_mean_income_df.rename(
                            columns={last_mean_income_df.index.name: "mundissec"},
                            inplace=True,
                        )
                last_mean_income_df["mundissec"] = last_mean_income_df[
                    "mundissec"
                ].astype(
                    str
                )  # Ensure data type matches

                # Merge on 'mundissec' while keeping timeseries_df's original index
                aux_timeseries_df = timeseries_df.copy()
                aux_timeseries_df.reset_index(inplace=True, drop=True)
                merged_df = pd.merge(
                    aux_timeseries_df,
                    last_mean_income_df,
                    on="mundissec",
                    how="left",
                )

                # Ensure that timeseries_df index is carried over
                merged_df.set_index(timeseries_df.index, inplace=True)
                timeseries_df = merged_df

            if self.add_socioeconomic_index:
                max_year = max(timeseries_df.index.map(extract_year))
                years_socioeconomic_index = [
                    int(re.search(r"\d+", col).group())
                    for col in self.socioeconomic_index_df.columns
                    if re.search(r"\d+", col) is not None
                ]
                max_year = min(max_year, max(years_socioeconomic_index))
                max_year_columns = [
                    col
                    for col in self.socioeconomic_index_df.columns
                    if str(max_year) in col
                ]
                # Resulting DataFrame with only columns that contain the max year
                last_socioeconomic_index_df = self.socioeconomic_index_df[
                    max_year_columns
                ]

                # Rename column "socioeconomic_index_{max_year}" to "{max_year}_ist"
                last_socioeconomic_index_df.rename(
                    columns={
                        f"valor_Índex socioeconòmic territorial_{max_year}": f"{max_year}_ist"
                    },
                    inplace=True,
                )

                # Extract 'mundissec' from the index of timeseries_df
                # This lambda function splits the index string at the underscore and takes the second part,
                # which is 'mundissec'
                timeseries_df["mundissec"] = timeseries_df.index.to_series().apply(
                    lambda x: x.split("_")[1]
                )

                # Ensure data types are compatible
                timeseries_df["mundissec"] = timeseries_df["mundissec"].astype(str)

                # If last_socioeconomic_index_df uses 'mundissec' as its index,
                # reset this index and ensure it's named correctly
                if (
                    last_socioeconomic_index_df.index.name == "mundissec"
                    or "mundissec" in last_socioeconomic_index_df.columns
                ):
                    last_socioeconomic_index_df.reset_index(inplace=True)
                    if last_socioeconomic_index_df.index.name == "mundissec":
                        last_socioeconomic_index_df.rename(
                            columns={
                                last_socioeconomic_index_df.index.name: "mundissec"
                            },
                            inplace=True,
                        )
                last_socioeconomic_index_df["mundissec"] = last_socioeconomic_index_df[
                    "mundissec"
                ].astype(
                    str
                )  # Ensure data type matches

                # Merge on 'mundissec' while keeping timeseries_df's original index
                aux_timeseries_df = timeseries_df.copy()
                aux_timeseries_df.reset_index(inplace=True, drop=True)
                merged_df = pd.merge(
                    aux_timeseries_df,
                    last_socioeconomic_index_df,
                    on="mundissec",
                    how="left",
                )

                # Ensure that timeseries_df index is carried over
                merged_df.set_index(timeseries_df.index, inplace=True)
                timeseries_df = merged_df

            if "mundissec" in timeseries_df.columns:
                timeseries_df = timeseries_df.drop(columns=["mundissec"])

            save_data(
                timeseries_df,
                f"../data/output/timeseries_{self.start_year}_{self.end_year}_{self.n_important_parties}_{self.add_election_type}_{self.add_born_abroad}_{self.add_age_groups}_{self.add_mean_income}_{self.add_socioeconomic_index}",
                index=True,
            )

        only_votes_df = merged_gdf.drop(columns=["geometry"])
        save_data(
            only_votes_df,
            f"../data/output/only_votes_{self.start_year}_{self.end_year}_{self.n_important_parties}",
        )
        logging.info("Saving output data with only past votes data done")

        # Apply the function to all columns in the GeoDataFrame
        column_mapping = shorten_column_names(merged_gdf.columns)
        # Rename the columns in the GeoDataFrame
        merged_gdf = merged_gdf.rename(columns=column_mapping)
        os.makedirs("../output/spatial", exist_ok=True)
        merged_gdf.to_file(
            f"../data/output/spatial/spatial_{self.start_year}_{self.end_year}_{self.n_important_parties}.shp"
        )

        logging.info("Data transformation done")
