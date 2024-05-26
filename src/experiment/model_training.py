"""
Module for training and evaluating machine learning models.
"""

from statistics import LinearRegression
from typing import Dict, Any

from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor


class ModelTraining:
    """Class for training and evaluating machine learning models."""

    def __init__(self, model_params: Dict[str, Any], hyperparams: Dict[str, Any]):
        """
        Initialize the ModelTraining object.

        Args:
            model_params (Dict[str, Any]): Parameters for the model.
            hyperparams (Dict[str, Any]): Hyperparameters for the model.
        """
        self.model_params = model_params
        self.hyperparams = hyperparams
        self.model = self.initialize_model()

    def initialize_model(self):
        """
        Initialize the machine learning model based on the specified model type.

        Returns:
            Any: Initialized machine learning model.

        Raises:
            ValueError: If the specified model type is not supported.
        """
        model_type = self.model_params["type"]
        if model_type == "xgboost":
            model = XGBRegressor(**self.hyperparams)
        elif model_type == "linear_regression":
            model = LinearRegression(**self.hyperparams)
        else:
            raise ValueError(f"Model type {model_type} is not supported.")
        return model

    def train(self, X_train, y_train):
        """
        Train the machine learning model.

        Args:
            X_train: Training data.
            y_train: Target values for the training data.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the machine learning model.

        Args:
            X_test: Test data.
            y_test: Target values for the test data.

        Returns:
            float: Accuracy of the model predictions.
        """
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
