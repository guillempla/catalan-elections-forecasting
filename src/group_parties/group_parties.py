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
    party_pairs = []

    for row in tqdm(similar_parties_matrix.index, desc="Extracting true pairs"):
        for col in similar_parties_matrix.columns:
            if row != col and similar_parties_matrix.at[row, col]:
                # Append the pair to the list if the parties are different
                party_pairs.append({"party_1": row, "party_2": col})

    # Convert the list of dictionaries to a DataFrame
    parties_table = pd.DataFrame(party_pairs, columns=["party_1", "party_2"])

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
        column_name: str = "clean_party_name",
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
        - column_name: str, optional
            The name of the column in the DataFrame that contains the party names.
            Defaults to "clean_party_name".
        - exclude_competed_together: bool, optional
            Whether to exclude parties that have competed together in the same election.
            Defaults to True.
        """
        self.df = load_data(clean_data_filename)
        self.output_filename = output_filename
        self.distance_function = distance_function
        self.threshold = threshold
        self.column_name = column_name
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
        distance_matrix = self._calculate_distance_matrix()
        boolean_distance_matrix = distance_matrix < self.threshold

        if self.exclude_competed_together:
            self._initialize_participated_in()

        party_similar_parties = extract_true_pairs(boolean_distance_matrix)

        self._initialize_similar_parties(party_similar_parties)

        # Create an ordered set of parties that have not been grouped yet
        # The set is in descending order by the number of total votes
        # each party received in every election
        not_grouped_parties = SortedSet(self.parties_dict.values())

        self._join_similar_parties(not_grouped_parties)

        # print(not_grouped_parties)

        # save_data(final_df, self.output_filename)

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
                votes=row["vots"],
            )
            for _, row in aggregated_df.iterrows()
        }
        return party_dict

    def _initialize_participated_in(self) -> None:
        """
        Initialize the participated_in attribute for each Party object.

        Returns:
        - None
        """
        # Iterate through each row in the DataFrame
        for row in self.df.itertuples():
            party_code = row.party_code
            election_name = row.nom_eleccio
            censal_section_id = row.mundissec
            party = self.parties_dict[party_code]
            # Add the election name and censal section ID to the participated_in set
            party.participated_in.add((election_name, censal_section_id))
            party.group_participated_in.add((election_name, censal_section_id))

    def _calculate_distance_matrix(self) -> pd.DataFrame:
        """
        Calculate a distance matrix for a list of strings using a distance algorithm.

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
            for j in range(i + 1, n):
                party_name_j = self.parties_dict[self.party_codes[j]].clean_name
                distance_matrix[i, j] = self.distance_function(
                    party_name_i, party_name_j
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
        while len(not_grouped_parties) > 0:
            # Get the last party in the set, which is the one with the most votes
            party = not_grouped_parties.pop()
            # print("Original Party: ", party)

            if party.joined:
                continue

            unjoined_similar_parties = SortedSet(party.similar_parties.values())
            # print("Similar Parties: ", unjoined_similar_parties)
            while len(unjoined_similar_parties) > 0:
                party_to_join = unjoined_similar_parties.pop()
                unjoined_similar_parties.union(party_to_join.similar_parties.values())
                # print("Parties left to join: ", party_to_join)
                party.join_parties(party_to_join)
