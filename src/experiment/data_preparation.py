from typing import Dict, Any

import pandas as pd

from utils.rw_files import load_data


class DataPreparation:
    def __init__(self, dataset_params: Dict[str, Any]):
        self.dataset_params = dataset_params
        self.dataset_filename = dataset_params["filename"]
        self.df = pd.DataFrame()

    def load_data(self):
        self.df = load_data(self.dataset_filename)
        self.df = self.sort_by_year_repetition(self.df)
        self.df = self.add_shifted_columns_grouped(self.df)

    @staticmethod
    def sort_by_year_repetition(df):
        # Extract election id components from the index
        df["year"] = df.index.map(
            lambda x: int(x.split("_")[0][1:5])
        )  # Assumes year is four digits long
        df["repetitionid"] = df.index.map(
            lambda x: int(x.split("_")[0][5:])
        )  # Assumes repetitionid immediately follows year

        # Sort the DataFrame by year and then by repetitionid
        df_sorted = df.sort_values(by=["year", "repetitionid"])

        # Drop the temporary columns used for sorting
        df_sorted = df_sorted.drop(columns=["year", "repetitionid"])

        return df_sorted

    @staticmethod
    def add_shifted_columns_grouped(df):
        # Extract 'mundissec' from the index
        df["mundissec"] = df.index.map(lambda x: x.split("_")[1])

        # Identify unique party codes by splitting each column name
        party_codes = set(
            col.split("_")[-1]
            for col in df.columns
            if ("_" in col) and (col.split("_")[-1].isupper())
        )

        # Iterate over each party code to create shifted columns group-wise
        for party_code in party_codes:
            # Identify columns for the current party code
            party_columns = [col for col in df.columns if col.endswith(party_code)]
            for col in party_columns:
                # Create a new shifted column name
                shifted_col_name = f"{col}_shifted"
                # Group by 'mundissec' and shift within each group
                df[shifted_col_name] = df.groupby("mundissec")[col].shift(
                    -1
                )  # Shift within each group

        # Drop the temporary 'mundissec' column after shifting
        df.drop(columns="mundissec", inplace=True)

        return df

    def split_data(self):
        # Identify unique elections from the index
        elections = self.df.index.map(lambda x: x.split("_")[0]).unique()

        # Latest and penultimate election identifiers
        last_election = elections[-1]
        penultimate_election = elections[-2] if len(elections) > 1 else None

        # Split the DataFrame based on the election identifiers
        new_data = self.df.loc[
            self.df.index.map(lambda x: x.split("_")[0]) == last_election
        ]
        test_data = (
            self.df.loc[
                self.df.index.map(lambda x: x.split("_")[0]) == penultimate_election
            ]
            if penultimate_election
            else pd.DataFrame()
        )
        train_data = self.df.loc[
            ~self.df.index.map(lambda x: x.split("_")[0]).isin(
                [last_election, penultimate_election]
            )
        ]

        # Columns that are not shifted
        non_shifted_columns = [
            col for col in self.df.columns if not col.endswith("_shifted")
        ]
        # Columns that are shifted
        shifted_columns = [col for col in self.df.columns if col.endswith("_shifted")]

        # Creating new_data, X_test, y_test, X_train, y_train
        new_data = new_data[non_shifted_columns]
        X_test = test_data[non_shifted_columns]
        y_test = test_data[shifted_columns]
        X_train = train_data[non_shifted_columns]
        y_train = train_data[shifted_columns]

        # Convert the data to float
        X_train = X_train.astype(float)
        y_train = y_train.astype(float)
        X_test = X_test.astype(float)
        y_test = y_test.astype(float)
        new_data = new_data.astype(float)

        return X_train, y_train, X_test, y_test, new_data
