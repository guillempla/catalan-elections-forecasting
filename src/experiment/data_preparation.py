from typing import Dict, Any


class DataPreparation:
    def __init__(self, dataset_params: Dict[str, Any]):
        self.dataset_params = dataset_params

    def load_data(self):
        # Load your dataset based on dataset_params
        # Example:
        data = ...  # Implement data loading here
        target = ...  # Implement target loading here
        return data, target

    def split_data(self, data, target):
        X_train, X_test, y_train, y_test = ...  # Implement data splitting here
        return X_train, X_test, y_train, y_test
