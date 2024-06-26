"""
ExperimentRunner class to run the experiment with the given configuration.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict

from experiment.data_preparation import DataPreparation
from experiment.experiment_attributes import ExperimentAttributes
from experiment.model_training import ModelTraining
from utils.rw_files import sentence_to_snake

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - Experiment Manager - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ExperimentRunner:
    """
    Class to run experiments based on the provided experiment configuration.
    """

    def __init__(self, experiment_config: Dict):
        """
        Initialize the ExperimentRunner with the experiment configuration.

        Args:
            experiment_config (Dict): The experiment configuration.
        """
        logging.info("Initializing the Experiment Runner.")

        self.experiment_attributes = ExperimentAttributes(experiment_config)
        self.datasets_params = self.experiment_attributes.dataset_params
        self.model_training = ModelTraining(
            self.experiment_attributes.model_type,
            self.experiment_attributes.model_params,
            self.experiment_attributes.fit_params,
        )

    def run_experiment(self) -> Dict:
        """
        Run the experiment and return the results.

        Returns:
            Dict: The experiment results.
        """
        logging.info(
            "Running the experiment %s.", self.experiment_attributes.experiment_name
        )

        start_date = datetime.now()
        datasets_metrics = {}
        error = None

        for dataset_params in self.datasets_params:
            data_preparation = DataPreparation(dataset_params)
            dataset_metrics = self.run_single_experiment(
                data_preparation=data_preparation
            )
            datasets_metrics[data_preparation.name] = dataset_metrics

        end_date = datetime.now()
        results = {
            "experiment_name": self.experiment_attributes.experiment_name,
            "start_date": start_date,
            "end_date": end_date,
            "elapsed_time": end_date - start_date,
            "status": "finished",
            "error": error if error else "None",
            "dataset_metrics": datasets_metrics if datasets_metrics else "None",
        }

        if results:
            self.log_metrics(results, self.experiment_attributes.experiment_name)

        return results

    def run_single_experiment(self, data_preparation: DataPreparation) -> Dict:
        """
        Run a single experiment with the given data preparation.

        Args:
            data_preparation (DataPreparation): The data preparation object.

        Returns:
            Dict: The experiment results for the single experiment.
        """
        start_date = datetime.now()
        status = "not started"
        metrics = None
        error = None

        try:
            status = "loading data"
            data_preparation.load_data()

            status = "splitting data"
            X_train, y_train, X_test, y_test, _ = data_preparation.split_data()

            status = "training model"
            if self.experiment_attributes.model_type == "xgboost":
                self.model_training.train(X_train, y_train, X_test, y_test)
            else:
                # If nan values are present in the data, fill them with 0
                X_train.fillna(0, inplace=True)
                self.model_training.train(X_train, y_train)

            status = "evaluating model"
            if self.experiment_attributes.model_type == "xgboost":
                metrics = self.model_training.evaluate(X_test, y_test)
            else:
                X_test.fillna(0, inplace=True)
                metrics = self.model_training.evaluate(X_test, y_test)
            status = "finished"

        except Exception as e:
            error = e
            status = f"error during {status}"

        end_date = datetime.now()

        results = {
            "dataset_name": data_preparation.name,
            "start_date": start_date,
            "end_date": end_date,
            "elapsed_time": end_date - start_date,
            "status": status,
            "error": error if error else "None",
            "params": {
                "model_type": self.experiment_attributes.model_type,
                "model_params": self.experiment_attributes.model_params,
                "fit_params": self.experiment_attributes.fit_params,
            },
            "metrics": metrics if metrics else "None",
        }

        return results

    @staticmethod
    def log_metrics(metrics: Dict, filename: str = "experiment_results"):
        """
        Log the experiment metrics to a file.

        Args:
            metrics (Dict): The experiment metrics.
            filename (str, optional): The filename for the log file.
                Defaults to "experiment_results".
        """
        folder_name = datetime.now().strftime("%Y%m%d_%H%M")
        folder_path = f"../results/{folder_name}"
        os.makedirs(folder_path, exist_ok=True)

        with open(
            f"{folder_path}/{sentence_to_snake(filename)}.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(metrics, fp, indent=4, sort_keys=True, default=str)
