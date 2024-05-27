"""
ExperimentRunner class to run the experiment with the given configuration.
"""

import csv
from datetime import datetime
from typing import Dict
from experiment.data_preparation import DataPreparation
from experiment.experiment_attributes import ExperimentAttributes
from experiment.model_training import ModelTraining


class ExperimentRunner:
    def __init__(self, experiment_config: Dict):
        self.experiment_attributes = ExperimentAttributes(experiment_config)
        self.data_preparation = DataPreparation(
            self.experiment_attributes.dataset_params
        )
        self.model_training = ModelTraining(
            self.experiment_attributes.model_params,
            self.experiment_attributes.hyperparams,
        )

    def run_experiment(self):
        start_date = datetime.now()
        status = "not started"
        metrics = None
        error = None

        try:
            status = "loading data"
            self.data_preparation.load_data()

            status = "splitting data"
            X_train, y_train, X_test, y_test, _ = self.data_preparation.split_data()

            status = "training model"
            self.model_training.train(X_train, y_train, X_test, y_test)

            status = "evaluating model"
            metrics = self.model_training.evaluate(X_test, y_test)
            status = "completed"
            end_date = datetime.now()

        except Exception as e:
            error = e
            status = f"error during {status}"

        if metrics:
            self.log_metrics(metrics)

        results = {
            "experiment_name": self.experiment_attributes.experiment_name,
            "start_date": start_date,
            "end_date": end_date,
            "elapsed_time": end_date - start_date,
            "status": status,
            "metrics": metrics if metrics else "None",
            "error": error if error else "None",
        }

        return results

    @staticmethod
    def log_metrics(metrics: Dict):
        with open(
            "experiment_metrics.csv", mode="a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(metrics.values())
