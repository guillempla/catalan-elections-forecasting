from typing import Dict, Any

import pandas as pd

from utils.rw_files import load_data


class DataPreparation:
    def __init__(self, dataset_params: Dict[str, Any]):
        self._name = dataset_params.get("name")
        self._path = dataset_params.get("path")
        self.df = pd.DataFrame()

    def load_data(self):
        self.df = load_data(self._path)
        self.df = self.sort_by_year_repetition(self.df)
        self.df = self.add_shifted_columns_grouped(self.df)
        self.df = self.remove_rows_with_null_sets(self.df)

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path

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
    def add_shifted_columns_grouped(df, num_shifts=1):
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
                for shift in range(1, num_shifts + 1):
                    # Create a new shifted column name
                    shifted_col_name = f"{col}_shifted_{shift}"
                    # Group by 'mundissec' and shift within each group
                    df[shifted_col_name] = df.groupby("mundissec")[col].shift(
                        -shift
                    )  # Shift within each group

        # Drop the temporary 'mundissec' column after shifting
        df.drop(columns="mundissec", inplace=True)

    @staticmethod
    def remove_rows_with_null_sets(df, max_null_sets=1):
        # Identify shifted columns
        shifted_columns = [col for col in df.columns if "shifted" in col]

        # Create a mask to identify rows with more than the allowed number of null sets
        null_sets_count = df[shifted_columns].isnull().sum(axis=1)
        mask = null_sets_count <= max_null_sets

        # Filter the DataFrame based on the mask
        filtered_df = df[mask]

        return filtered_df

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

        # Identify shifted columns and non-shifted columns
        shifted_columns = [col for col in self.df.columns if "_shifted_" in col]
        non_shifted_columns = [
            col for col in self.df.columns if col not in shifted_columns
        ]

        # Identify the highest shift value
        max_shift_value = max(int(col.split("_")[-1]) for col in shifted_columns)

        # Columns with the highest shift value
        y_shifted_columns = [
            col
            for col in shifted_columns
            if col.endswith(f"_shifted_{max_shift_value}")
        ]
        # Columns with other shift values
        x_shifted_columns = [
            col for col in shifted_columns if col not in y_shifted_columns
        ]

        # Ensure new_data only contains non-shifted columns
        new_data = new_data[non_shifted_columns]

        # Split test and train data into X (non-shifted and lower shift values) and y (highest shift values)
        X_test = test_data[non_shifted_columns + x_shifted_columns]
        y_test = test_data[y_shifted_columns]
        X_train = train_data[non_shifted_columns + x_shifted_columns]
        y_train = train_data[y_shifted_columns]

        return X_train, y_train, X_test, y_test, new_data
