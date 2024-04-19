"""
Group parties and save it as a CSV or Picke file.
"""

from typing import List, Union
import logging
import pandas as pd
import numpy as np
import textdistance
from tqdm import tqdm
from sortedcontainers import SortedSet
from group_parties.party import Party
from utils.rw_files import load_data, save_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def most_common(series):
    modes = series.mode()
    if not modes.empty:
        return modes[0]
    return None


def extract_true_pairs(similar_parties_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Extract pairs of parties with True values from a similarity matrix.

    Parameters:
    - similar_parties_matrix: DataFrame, a matrix indicating similarity (True)
      between pairs of parties.

    Returns:
    - DataFrame with two columns (party_1, party_2) listing all pairs of parties with True values.
    """
    # Use `stack` to convert the DataFrame into a Series where multi-index represents (row, col),
    # and then filter by True values.
    true_pairs = similar_parties_matrix.stack()[lambda x: x].index

    # Filter out pairs where row == col (same parties)
    true_pairs = [(idx[0], idx[1]) for idx in true_pairs if idx[0] != idx[1]]

    # Convert list of tuples to DataFrame
    parties_table = pd.DataFrame(true_pairs, columns=["party_1", "party_2"])

    return parties_table


class GroupParties:
    """
    Group data.
    """

    def __init__(
        self,
        clean_data_filename: str = "../data/processed/catalan-elections-clean-data.pkl",
        output_filename: str = "../data/processed/catalan-elections-grouped-data",
        distance_function: callable = textdistance.jaro_winkler.distance,
        threshold: float = 0.15,
        only_name: bool = False,
        only_abbr: bool = False,
        exclude_competed_together: bool = True,
    ) -> None:
        """
        Initialize the GroupParties class.

        Parameters:
        - clean_data_filename: str, optional
            The filename of the clean data file.
        - output_filename: str, optional
            The filename of the output file.
        - distance_function: str or callable, optional
            The distance function used to compare party names.
            It should be a distance function that takes two strings as input and returns a float.
            Defaults to textdistance.jaro_winkler.distance.
        - threshold: float, optional
            The threshold value used to determine if two party names are similar.
            Defaults to 0.2.
        - only_name: bool, optional
            Whether to use only the party name to calculate the distance.
            Or use the combination of party name and party abbreviature
            Defaults to False.
        - only_abbr: bool, optional
            Whether to use only the party abbreviature to calculate the distance.
            Or use the combination of party name and party abbreviature
            Defaults to False.
        - exclude_competed_together: bool, optional
            Whether to exclude parties that have competed together in the same election.
            Defaults to True.
        """
        self.df = load_data(clean_data_filename)
        self.output_filename = output_filename
        self.distance_function = distance_function
        self.threshold = threshold
        self.only_name = only_name
        self.only_abbr = only_abbr
        self.exclude_competed_together = exclude_competed_together
        self.parties_dict: dict[str, Party] = self._initialize_parties_dict()
        self.party_codes = list(self.parties_dict.keys())

    def group_parties(self) -> None:
        """
        Joins parties in the df dataframe based on the distance_function and threshold.

        Returns:
        - None
        """
        logging.info("Grouping parties codes.")
        distance_matrix = self._calculate_distance_matrix(
            only_name=self.only_name, only_abbr=self.only_abbr
        )
        boolean_distance_matrix = distance_matrix < self.threshold

        if self.exclude_competed_together:
            self._initialize_participated_in()

        party_similar_parties = extract_true_pairs(boolean_distance_matrix)

        self._initialize_similar_parties(party_similar_parties)

        # Create an ordered set of parties that have not been grouped yet
        # The set is in descending order by the number of total votes
        # each party received in every election
        not_grouped_parties = SortedSet(self.parties_dict.values())

        joined_parties = self._join_similar_parties(not_grouped_parties)

        save_data(joined_parties, "../data/processed/joined_parties")

        final_df = self.df.merge(
            joined_parties, how="left", left_on="party_code", right_on="party_code"
        )

        final_df["joined_code"] = np.where(
            final_df["joined_code"].isnull(),
            final_df["party_code"],
            final_df["joined_code"],
        )

        final_df["joined_name"] = np.where(
            final_df["joined_name"].isnull(),
            final_df["party_name"],
            final_df["joined_name"],
        )

        final_df["joined_abbr"] = np.where(
            final_df["joined_abbr"].isnull(),
            final_df["party_abbr"],
            final_df["joined_abbr"],
        )

        final_df["joined_clean_name"] = np.where(
            final_df["joined_clean_name"].isnull(),
            final_df["clean_party_name"],
            final_df["joined_clean_name"],
        )

        final_df["joined_clean_abbr"] = np.where(
            final_df["joined_clean_abbr"].isnull(),
            final_df["clean_party_abbr"],
            final_df["joined_clean_abbr"],
        )

        final_df["joined_color"] = np.where(
            final_df["joined_color"].isnull(),
            final_df["party_color"],
            final_df["joined_color"],
        )

        save_data(final_df, self.output_filename)

    def _initialize_parties_dict(self) -> dict[str, Party]:
        """
        Initialize a dictionary with party names as keys and Party objects as values.

        Returns:
        - dict[str, Party], a dictionary with party names as keys and Party objects as values.
        """
        # Aggregate the dataframe using custom function for mode
        aggregated_df = (
            self.df.groupby("party_code")
            .agg(
                {
                    "party_name": most_common,  # Most common name
                    "party_abbr": most_common,
                    "clean_party_name": most_common,
                    "clean_party_abbr": most_common,
                    "party_color": most_common,
                    "vots": "sum",  # Summing votes for each party code
                }
            )
            .reset_index()
        )

        # Create a dictionary of Party objects
        party_dict = {
            row["party_code"]: Party(
                party_name=row["party_name"],
                party_abbr=row["party_abbr"],
                party_code=row["party_code"],
                party_clean_name=row["clean_party_name"],
                party_clean_abbr=row["clean_party_abbr"],
                party_color=row["party_color"],
                votes=row["vots"],
            )
            for _, row in aggregated_df.iterrows()
        }
        return party_dict

    def _initialize_participated_in(self) -> None:
        """
        Initialize the participated_in attribute for each Party object using optimized pandas operations.

        Returns:
        - None
        """
        # Group by 'party_code' and aggregate unique pairs of 'nom_eleccio' and 'mundissec' into a set
        group_data = self.df.groupby("party_code").apply(
            lambda x: set(zip(x.nom_eleccio, x.mundissec))
        )

        # Update parties_dict directly with the computed sets
        for party_code, participated_set in group_data.items():
            party = self.parties_dict[party_code]
            party.participated_in.update(participated_set)
            party.group_participated_in.update(participated_set)

    def _calculate_distance_matrix(
        self, only_name=False, only_abbr=False
    ) -> pd.DataFrame:
        """
        Calculate a distance matrix for a list of strings using a distance algorithm.

        Parameters:
        - only_name: bool, optional
            Whether to use only the party name to calculate the distance.
            Or use the combination of party name and party abbreviature
            Defaults to False.

        Returns:
        - pd.DataFrame, a distance matrix.
        """
        n = len(self.party_codes)

        distance_matrix = np.zeros(
            (n, n), dtype=float
        )  # Initialize a matrix with zeros
        max_distance = 100.0

        # Only calculate for the upper half of the matrix
        for i in tqdm(range(n), desc="Calculating distances"):
            party_name_i = self.parties_dict[self.party_codes[i]].clean_name
            party_abbr_i = self.parties_dict[self.party_codes[i]].clean_abbr
            for j in range(i + 1, n):
                party_name_j = self.parties_dict[self.party_codes[j]].clean_name
                party_abbr_j = self.parties_dict[self.party_codes[j]].clean_abbr

                if only_name:
                    distance_matrix[i, j] = self.distance_function(
                        party_name_i, party_name_j
                    )
                elif only_abbr:
                    distance_matrix[i, j] = self.distance_function(
                        party_abbr_i, party_abbr_j
                    )
                else:
                    distance_matrix[i, j] = self.distance_function(
                        party_name_i + party_abbr_i, party_name_j + party_abbr_j
                    )
                distance_matrix[j, i] = max_distance

        # Making sure the diagonal is high to simulate distance to itself (will be filtered out)
        np.fill_diagonal(distance_matrix, 100.0)

        # Convert the NumPy array to a pandas DataFrame
        return pd.DataFrame(
            distance_matrix, index=self.party_codes, columns=self.party_codes
        )

    def _initialize_similar_parties(
        self, similar_parties: List[Union[str, List[str]]]
    ) -> None:
        """
        Set similar parties for each party in the parties_dict.

        Parameters:
        - similar_parties: List[Union[str, List[str]]], a list of similar party names.
        """
        for row in similar_parties.itertuples():
            party_1 = self.parties_dict[row.party_1]
            party_2 = self.parties_dict[row.party_2]
            party_1.add_similar_party(party_2)
            party_2.add_similar_party(party_1)

    def _join_similar_parties(self, not_grouped_parties) -> None:
        joined_parties_df = pd.DataFrame(
            columns=[
                "party_code",
                "joined_code",
                "joined_name",
                "joined_abbr",
                "joined_clean_name",
                "joined_clean_abbr",
                "joined_color",
            ]
        )

        while len(not_grouped_parties) > 0:
            # Get and remove the last party in the set, which is the one with the most votes
            party = not_grouped_parties.pop()

            if party.joined:
                continue

            # Get the similar parties of the current party and sort them by votes
            unjoined_similar_parties = SortedSet(party.similar_parties.values())
            while len(unjoined_similar_parties) > 0:
                # Get, remove and group the last party in the set
                party_to_join = unjoined_similar_parties.pop()
                unjoined_similar_parties.union(party_to_join.similar_parties.values())
                are_joined = party.join_parties(party_to_join)

                # Use concat instead of append
                if are_joined:
                    # Create a temporary DataFrame to hold the new row
                    new_row = pd.DataFrame(
                        {
                            "party_code": [party_to_join.code],
                            "joined_code": [party.code],
                            "joined_name": [party.name],
                            "joined_abbr": [party.abbr],
                            "joined_clean_name": [party.clean_name],
                            "joined_clean_abbr": [party.clean_abbr],
                            "joined_color": [party.color],
                        }
                    )
                    joined_parties_df = pd.concat(
                        [joined_parties_df, new_row], ignore_index=True
                    )

        return joined_parties_df
