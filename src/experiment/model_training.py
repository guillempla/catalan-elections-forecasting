"""
Module for training and evaluating machine learning models.
"""

import numpy as np
import pandas as pd
from statistics import LinearRegression
from typing import Dict, Any

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


class ModelTraining:
    """Class for training and evaluating machine learning models."""

    def __init__(
        self, model_type: str, model_params: Dict[str, Any], fit_params: Dict[str, Any]
    ):
        """
        Initialize the ModelTraining object.

        Args:
            model_params (Dict[str, Any]): Parameters for the model.
            fit_params (Dict[str, Any]): Hyperparameters for the model.
        """
        self.model_params = model_params
        self.fit_params = fit_params
        self.model_type = model_type
        self.model = self.initialize_model()

    def initialize_model(self):
        """
        Initialize the machine learning model based on the specified model type.

        Returns:
            Any: Initialized machine learning model.

        Raises:
            ValueError: If the specified model type is not supported.
        """
        if self.model_type == "xgboost":
            model = XGBRegressor(**self.model_params)
        elif self.model_type == "linear_regression":
            model = LinearRegression(**self.model_params)
        elif self.model_type == "knn":
            model = KNeighborsRegressor(**self.model_params)
        else:
            raise ValueError(f"Model type {self.model_type} is not supported.")
        return model

    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train the machine learning model.

        Args:
            X_train: Training data.
            y_train: Target values for the training data.
        """

        if X_test is None or y_test is None:
            if self.fit_params is None:
                self.model.fit(X_train, y_train)
            else:
                self.model.fit(X_train, y_train, **self.fit_params)
        else:
            eval_set = [(X_test, y_test)]
            if self.fit_params is None:
                self.model.fit(X_train, y_train, eval_set=eval_set)
            else:
                self.model.fit(X_train, y_train, eval_set=eval_set, **self.fit_params)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the machine learning model.

        Args:
            X_test: Test data.
            y_test: Target values for the test data.

        Returns:
            float: Accuracy of the model predictions.
        """
        # Initialize a dictionary to store metrics for each column
        metrics = {}

        if self.model_type == "xgboost":
            predictions = self.model.predict(
                X_test, iteration_range=(0, self.model.best_iteration + 1)
            )
            if not isinstance(predictions, pd.DataFrame):
                predictions = pd.DataFrame(
                    predictions, index=y_test.index, columns=y_test.columns
                )

            # Loop through each column in y_test to calculate metrics
            for column in y_test.columns:
                mse = mean_squared_error(y_test[column], predictions[column])
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test[column], predictions[column])

                # Store metrics in the dictionary
                metrics[column] = {"RMSE": rmse, "R^2": r2}
        elif self.model_type == "knn":
            # Predict on the test set
            predictions = self.model.predict(X_test)
            if not isinstance(predictions, pd.DataFrame):
                predictions = pd.DataFrame(
                    predictions, index=y_test.index, columns=y_test.columns
                )

            # Loop through each column in y_test to calculate metrics
            for column in y_test.columns:
                mse = mean_squared_error(y_test[column], predictions[column])
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test[column], predictions[column])

                # Store metrics in the dictionary
                metrics[column] = {"RMSE": rmse, "R^2": r2}
        else:
            raise ValueError(
                f"Evaluation for model type {self.model_type} is not supported."
            )

        return metrics
