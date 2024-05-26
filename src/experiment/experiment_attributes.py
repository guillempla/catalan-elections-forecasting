"""
This module contains the ExperimentConfig class,
which represents the configuration for an experiment.
"""

from typing import Dict, Any


class ExperimentAttributes:
    """
    ExperimentConfig class represents the configuration for an experiment.
    """

    def __init__(self, experiment_config: Dict[str, Any]):
        """
        Initialize the ExperimentConfig object.

        Args:
            experiment_config (Dict[str, Any]): A dictionary containing the configuration
                for the experiment.
        """
        self.experiment_name = experiment_config.get("experiment_name")
        self.dataset_params = experiment_config.get("dataset_params")
        self.model_params = experiment_config.get("model_type_params")
        self.hyperparams = experiment_config.get("hyperparams")

    @property
    def experiment_name(self):
        """
        Get the name of the experiment.

        Returns:
            str: The name of the experiment.
        """
        return self._experiment_name

    @experiment_name.setter
    def experiment_name(self, value):
        """
        Set the name of the experiment.

        Args:
            value (str): The name of the experiment.
        """
        self._experiment_name = value

    @property
    def dataset_params(self):
        """
        Get the parameters for the dataset.

        Returns:
            Dict[str, Any]: A dictionary containing the parameters for the dataset.
        """
        return self._dataset_params

    @dataset_params.setter
    def dataset_params(self, value):
        """
        Set the parameters for the dataset.

        Args:
            value (Dict[str, Any]): A dictionary containing the parameters for the dataset.
        """
        self._dataset_params = value

    @property
    def model_params(self):
        """
        Get the parameters for the model.

        Returns:
            Dict[str, Any]: A dictionary containing the parameters for the model.
        """
        return self._model_params

    @model_params.setter
    def model_params(self, value):
        """
        Set the parameters for the model.

        Args:
            value (Dict[str, Any]): A dictionary containing the parameters for the model.
        """
        self._model_params = value

    @property
    def hyperparams(self):
        """
        Get the hyperparameters for the experiment.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters for the experiment.
        """
        return self._hyperparams

    @hyperparams.setter
    def hyperparams(self, value):
        """
        Set the hyperparameters for the experiment.

        Args:
            value (Dict[str, Any]): A dictionary containing the hyperparameters for the experiment.
        """
        self._hyperparams = value
