"""
This module contains the ExperimentConfig class, which represents the configuration for an experiment.
"""

from typing import Dict, Any


class ExperimentConfig:
    """
    ExperimentConfig class represents the configuration for an experiment.

    Attributes:
        dataset_params (Dict[str, Any]): A dictionary containing the parameters for the dataset.
        model_type_params (Dict[str, Any]): A dictionary containing the parameters for the model.
        hyperparams (Dict[str, Any]): A dictionary containing the hyperparameters for the
            experiment.
    """

    class ExperimentConfig:
        def __init__(
            self,
            dataset_params: Dict[str, Any],
            model_type_params: Dict[str, Any],
            hyperparams: Dict[str, Any],
        ):
            """
            Initialize the ExperimentConfig object.

            Args:
                dataset_params (Dict[str, Any]): A dictionary containing the parameters for the dataset.
                model_type_params (Dict[str, Any]): A dictionary containing the parameters for the model.
                hyperparams (Dict[str, Any]): A dictionary containing the hyperparameters for the experiment.
            """
            self.dataset_params = dataset_params
            self.model_params = model_type_params
            self.hyperparams = hyperparams
