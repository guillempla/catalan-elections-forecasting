"""
ExperimentRunner class to run the experiment with the given configuration.
"""

import json
from datetime import datetime
from typing import Dict
from experiment.data_preparation import DataPreparation
from experiment.experiment_attributes import ExperimentAttributes
from experiment.model_training import ModelTraining
from utils.rw_files import sentence_to_snake


class ExperimentRunner:
    def __init__(self, experiment_config: Dict):
        self.experiment_attributes = ExperimentAttributes(experiment_config)
        self.datasets_params = self.experiment_attributes.dataset_params
        self.model_training = ModelTraining(
            self.experiment_attributes.model_type,
            self.experiment_attributes.model_params,
            self.experiment_attributes.fit_params,
        )

    def run_experiment(self):
        start_date = datetime.now()
        status = "not started"
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
            "status": status,
            "error": error if error else "None",
            "dataset_metrics": datasets_metrics if datasets_metrics else "None",
        }

        if results:
            self.log_metrics(results, self.experiment_attributes.experiment_name)

        return results

    def run_single_experiment(self, data_preparation: DataPreparation) -> Dict:
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
            self.model_training.train(X_train, y_train, X_test, y_test)

            status = "evaluating model"
            metrics = self.model_training.evaluate(X_test, y_test)
            status = "completed"

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
            "metrics": metrics if metrics else "None",
        }

        return results

    @staticmethod
    def log_metrics(metrics: Dict, filename: str = "experiment_results"):
        filename = (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + sentence_to_snake(filename)
        )
        with open(f"../results/{filename}.json", "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=4, sort_keys=True, default=str)
