"""
Transform censal sections data and results data into a single output dataframe
"""

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
        start_year: int = 2008,
        end_year: int = 2024,
        n_important_parties: int = 9,
        transform_to_timeseries: bool = False,
        add_election_type: int = 0,  # 0: no election type, 1: add election type as one column, 2: add election type as one column per election type
        add_born_abroad: bool = False,
        add_mean_income: bool = False,
        add_age_groups: bool = False,
    ) -> None:
        """
        Initialize the class with the paths to the censal sections and results data

        Args:
            censal_sections_path (str): Path to the censal sections data file
            results_path (str): Path to the election results data file
            output_path (str, optional): Path to save the output dataframe. Defaults to "../data/output/censal_sections_results".
        """
        self.censal_sections_gdf = load_data(censal_sections_path)
        self.results_df = load_data(results_path)
        self.born_abroad_df = load_data(born_abroad_path)
        self.output_path = output_path
        self.start_year = start_year
        self.end_year = end_year
        self.n_important_parties = n_important_parties
        self.transform_to_timeseries = transform_to_timeseries
        self.add_election_type = add_election_type
        self.add_born_abroad = add_born_abroad
        self.add_mean_income = add_mean_income
        self.add_age_groups = add_age_groups

    def transform_data(self) -> None:
        """
        Transform censal sections data and results data into a single output dataframe
        """

        # filter the results_df to only include the years between start_year and end_year
        self.results_df = self.results_df[
            self.results_df["year"].astype(int).between(self.start_year, self.end_year)
        ]

        # # fix cup-g and cup-pr joined_code
        # self.results_df.loc[
        #     (self.results_df["party_abbr"] == "CUP-G")
        #     | (self.results_df["party_abbr"] == "CUP-PR"),
        #     "joined_code",
        # ] = 1003
        # # fix cup-g and cup-pr joined_name
        # self.results_df.loc[
        #     (self.results_df["party_abbr"] == "CUP-G")
        #     | (self.results_df["party_abbr"] == "CUP-PR"),
        #     "joined_name",
        # ] = "Candidatura d'Unitat Popular"
        # # fix cup-g and cup-pr joined_abbr
        # self.results_df.loc[
        #     (self.results_df["party_abbr"] == "CUP-G")
        #     | (self.results_df["party_abbr"] == "CUP-PR"),
        #     "joined_abbr",
        # ] = "CUP"

        self.results_df = manually_group_parties(
            self.results_df,
            {
                1013: [71],
                1031: [12, 1007, 1000],
                1003: [2015673, 82064190],
                # 999999999: [
                #     5000000,
                #     170564112,
                #     430244110,
                #     895,
                #     860,
                #     80655190,
                #     81654190,
                #     252094190,
                #     252484190,
                #     431704190,
                # ],
            },
        )

        important_parties_df = select_important_parties(
            self.results_df, n_important_parties=self.n_important_parties
        )
        # important_parties = select_important_parties_last_election(
        #     self.results_df, n_important_parties=self.n_important_parties
        # )

        # Pivot the DataFrame
        logging.info("Pivoting the DataFrame")
        results_wide_df = important_parties_df.pivot_table(
            index="mundissec",
            columns=["joined_code", "id_eleccio"],
            values=[
                "vots",
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

        # save_data(results_wide_df, "../data/output/results_wide_df")

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

                # Step 1: Extract 'mundissec' from the index of timeseries_df
                # This lambda function splits the index string at the underscore and takes the second part, which is 'mundissec'
                timeseries_df["mundissec"] = timeseries_df.index.to_series().apply(
                    lambda x: x.split("_")[1]
                )

                # Step 2: Ensure data types are compatible
                timeseries_df["mundissec"] = timeseries_df["mundissec"].astype(str)

                # If last_born_abroad_df uses 'mundissec' as its index, reset this index and ensure it's named correctly
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

                # Step 3: Merge on 'mundissec' while keeping timeseries_df's original index
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

            save_data(
                timeseries_df,
                f"../data/output/timeseries_{self.start_year}_{self.end_year}_{self.n_important_parties}_{self.add_election_type}_{self.add_born_abroad}",
                index=True,
            )

        only_votes_df = merged_gdf.drop(columns=["geometry"])
        save_data(
            only_votes_df,
            f"../data/output/only_votes_{self.start_year}_{self.end_year}_{self.n_important_parties}",
        )

        logging.info("Saving output data with only past votes data done")

        # merged_gdf.to_file("../data/output/merged_data.geojson", driver="GeoJSON")
        logging.info("Data transformation done")
