"""
Group parties and save it as a CSV or Picke file.
"""

from typing import List, Dict

import logging
import pandas as pd
import numpy as np
import textdistance
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def extract_true_pairs(similar_parties_matrix):
    """
    Extract pairs of parties with True values from a similarity matrix.

    Parameters:
    - similar_parties_matrix: DataFrame, a matrix indicating similarity (True)
    between pairs of parties.

    Returns:
    - DataFrame with two columns (party_1, party_2) listing all pairs of parties with True values.
    """
    party_pairs = []

    for row in tqdm(similar_parties_matrix.index, desc="Extracting true pairs"):
        for col in similar_parties_matrix.columns:
            if row != col and similar_parties_matrix.at[row, col]:
                # Append the pair to the list if the parties are different
                party_pairs.append({"party_1": row, "party_2": col})

    # Convert the list of dictionaries to a DataFrame
    parties_table = pd.DataFrame(party_pairs, columns=["party_1", "party_2"])

    return parties_table


def add_most_voted_party_code_column(most_voted_matrix, similar_parties):
    """
    Adds a 'most_voted_party_code' column to the similar_parties dataframe.

    Parameters:
    - most_voted_matrix: DataFrame, dataframe matrix party columns.
    - similar_parties: DataFrame, DataFrame with two columns (party_1, party_2).

    Returns:
    - DataFrame, the original dataframe with an added 'most_voted_party_code' column.
    """
    similar_parties["most_voted_party_code"] = similar_parties.apply(
        lambda row: most_voted_matrix.at[row["party_1"], row["party_2"]], axis=1
    )
    return similar_parties


def get_party_code_most_votes(party_codes, party_codes_votes):
    """
    Given a list of party codes, return the party code with the most votes from a list of party codes.

    Parameters:
    - party_codes: list, a list of party codes.
    - party_codes_votes: DataFrame, a list of party codes with their votes.
    """
    try:
        max_code = party_codes_votes.loc[party_codes].idxmax()
    except IndexError:
        max_code = party_codes[0]

    return max_code


class GroupParties:
    """
    Group data.
    """

    def __init__(
        self,
        df,
        distance_function=None,
        threshold=None,
        column_name="clean_party_name",
        exclude_competed_together=True,
    ) -> None:
        """
        Initialize the GroupParties class.

        Parameters:
        - df: DataFrame
            The input DataFrame containing the party data.
        - distance_function: function, optional
            The distance function used to compare party names.
            If not provided, the default distance function will be used.
        - threshold: float, optional
            The threshold value used to determine if two party names are similar.
            If not provided, the default threshold will be used.
        - column_name: str, optional
            The name of the column in the DataFrame that contains the party names.
            Defaults to "clean_party_name".
        - exclude_competed_together: bool, optional
            Whether to exclude parties that have competed together in the same election.
            Defaults to True.
        """
        self.df = df
        self.distance_function = distance_function
        self.threshold = threshold
        self.column_name = column_name
        self.exclude_competed_together = exclude_competed_together

    def group_parties(self):
        """
        Joins parties in the df dataframe based on the distance_function and threshold.

        Parameters:
        - df: DataFrame, dataframe with party1 and party2 columns.
        - distance_function: function, a distance function from the textdistance library.
        - threshold: float, the threshold for the distance function.
        - column: str, the name of the column in the DataFrame containing the party strings.

        Returns:
        - DataFrame, the original dataframe with an added 'most_voted_party_code' column.
        """
        logging.info("Grouping parties codes.")
        # Filter the df to only include those parties that have competed in municipal elections
        # This is done to have the same list of parties as in the competed_together_matrix
        df_filtered = self.df[self.df["id_nivell_territorial"] == "MU"]
        party_names = sorted(df_filtered[self.column_name].unique().tolist())
        distance_matrix = self.calculate_distance_matrix(party_names)
        boolean_distance_matrix = distance_matrix < self.threshold
        most_voted_matrix = self.calculate_most_voted_party_code_matrix()

        if self.exclude_competed_together:
            competed_together_matrix = self.parties_competed_together_matrix()
            not_competed_together_matrix = ~competed_together_matrix
            similar_parties_matrix = (
                boolean_distance_matrix & not_competed_together_matrix
            )
        else:
            similar_parties_matrix = boolean_distance_matrix
        similar_parties = extract_true_pairs(similar_parties_matrix)
        similar_parties = add_most_voted_party_code_column(
            most_voted_matrix, similar_parties
        )

        # Combine both party columns and flatten the dataset
        # while keeping the 'most_voted_party_code'
        parties_flattened = (
            similar_parties.set_index("most_voted_party_code")
            .stack()
            .reset_index(name="party")
            .drop("level_1", axis=1)
            .drop_duplicates(subset=["party"])  # Removing duplicate party entries
        )
        # Creating a new dataset with unique party names
        # and their associated 'most_voted_party_code'
        unique_parties = parties_flattened.drop_duplicates(
            "party", keep="first"
        ).reset_index(drop=True)

        # Merge 'self.df' with 'unique_parties' on "clean_party_name" and "party"
        merged_df = pd.merge(
            self.df,
            unique_parties,
            how="left",
            left_on="clean_party_name",
            right_on="party",
        )

        # Create the "joined_code" self.column_name
        merged_df["joined_code"] = np.where(
            merged_df["most_voted_party_code"].isnull(),
            merged_df["party_code"],
            merged_df["most_voted_party_code"],
        )

        # Dropping unnecessary columns for clarity
        final_df = merged_df.drop(["party", "most_voted_party_code"], axis=1)
        return final_df

    def calculate_distance_matrix(self, party_names):
        """Calculate a distance matrix for a list of strings using a distance algorithm

        Set to max_distance for the diagonal and below the diagonal
        to avoid taking into account a value 2 times

        Args:
        party_names: A list with the party names (or abreviations) to calculate the distance matrix for
        distance_algorithm: A distance algorithm from the textdistance library
        column_name: The name of the column in the DataFrame containing the party strings
        """
        n = len(party_names)

        distance_matrix = np.zeros(
            (n, n), dtype=float
        )  # Initialize a matrix with zeros
        max_distance = 100.0

        # Only calculate for the upper half of the matrix
        for i in tqdm(range(n), desc="Calculating distances"):
            for j in range(i + 1, n):
                distance_matrix[i, j] = self.distance_function(
                    party_names[i], party_names[j]
                )
                distance_matrix[j, i] = max_distance

        # Making sure the diagonal is high to simulate distance to itself (will be filtered out)
        np.fill_diagonal(distance_matrix, 100.0)

        # Convert the NumPy array to a pandas DataFrame
        return pd.DataFrame(distance_matrix, index=party_names, columns=party_names)

    def get_party_codes_votes(self, territory="CA"):
        """
        Get the party codes and their sum of votes for a list of party names.

        Parameters:
        - df: DataFrame, the original dataframe with party information.
        - territory: str, the territorial level to filter the dataframe.
        """
        # Filter the df by id_nivell_territorial equal to territory
        filtered_df = self.df[self.df["id_nivell_territorial"] == territory]
        # Get the party_codes with their sum of votes
        party_codes_votes = filtered_df.groupby("party_code")["vots"].sum()
        return party_codes_votes

    def get_codes_from_names(self, df_grouped, party_name):
        """
        Get the party codes for a list of party names.

        Parameters:
        - df: DataFrame, the original dataframe with party information.
        - party_name: str, name of the party.
        - column: str, the column name used to identify parties.
        """
        return list(
            df_grouped[df_grouped[self.column_name] == party_name][
                "party_code"
            ].unique()
        )

    def get_party_codes_dict(self):
        """
        Get a dictionary of party names and their party codes using an optimized approach.

        Parameters:
        - df: DataFrame, the original dataframe with party information.
        - column: str, the column name used to identify parties.
        """
        # Group by the party name column and apply the get_codes_from_names function to each group.
        party_codes_dict = (
            self.df.groupby(self.column_name)
            .apply(lambda x: self.get_codes_from_names(x, x[self.column_name].iloc[0]))
            .to_dict()
        )
        return party_codes_dict

    def calculate_most_voted_party_code_matrix(self):
        """
        Calculate the most voted party code matrix for a dataframe.

        Parameters:
        - df: DataFrame, the original dataframe with party information.
        """
        # Get the party names
        df_filtered = self.df[self.df["id_nivell_territorial"] == "MU"]
        party_names = df_filtered["clean_party_name"].unique()

        # Precalculate the sum of votes for each party code
        party_codes_votes = self.get_party_codes_votes()
        # Precalculate the party codes for each party name
        party_codes_dict = self.get_party_codes_dict()

        # Create an empty boolean matrix
        matrix_size = len(party_names)
        most_voted_matrix = pd.DataFrame("", index=party_names, columns=party_names)

        # Iterate through each party name once
        for i in tqdm(
            range(matrix_size), desc="Calculating most voted party code matrix"
        ):
            party_name_i = party_names[i]
            codes_i = party_codes_dict[party_name_i]
            for j in range(i + 1, matrix_size):
                party_name_j = party_names[j]
                codes_j = party_codes_dict[party_name_j]

                combined_codes = list(set(codes_i + codes_j))
                max_party_code = get_party_code_most_votes(
                    combined_codes, party_codes_votes
                )
                most_voted_matrix.at[party_name_i, party_name_j] = most_voted_matrix.at[
                    party_name_j, party_name_i
                ] = max_party_code

        return most_voted_matrix

    def parties_competed_together_matrix(self):
        # Filter DataFrame for territorial level "MU"
        df_filtered = self.df[self.df["id_nivell_territorial"] == "MU"]

        # Group by party name and aggregate unique election identifiers into sets
        party_elections = df_filtered.groupby(self.column_name)["nom_eleccio"].apply(
            set
        )

        # Get a sorted list of unique party names for consistent ordering
        party_names = sorted(party_elections.index)

        # Create an empty boolean matrix
        matrix_size = len(party_names)
        bool_matrix = pd.DataFrame(False, index=party_names, columns=party_names)

        # Iterate through each party name once, showing progress with tqdm
        for i in tqdm(
            range(matrix_size), desc="Calculate parties that competed together"
        ):
            party_name_i = party_names[i]
            elections_i = party_elections[party_name_i]
            for j in range(i + 1, matrix_size):
                party_name_j = party_names[j]
                # Compare sets of election IDs for common elements
                # Set the cell to True if there's an intersection
                bool_matrix.at[party_name_i, party_name_j] = bool_matrix.at[
                    party_name_j, party_name_i
                ] = not elections_i.isdisjoint(party_elections[party_name_j])

        return bool_matrix
