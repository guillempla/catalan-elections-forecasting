"""
Module to execute multiple experiments and store the results.
"""

import logging
from typing import Dict, List
import pandas as pd

from experiment.experiment_runner import ExperimentRunner
from utils.rw_files import save_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - Experiment Manager - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ExperimentManager:
    """
    Class to manage and run experiments.

    Args:
        experiments (List[Dict]): A list of dictionaries representing the experiments to run.
    """

    def __init__(self, experiments: List[Dict]):
        logging.info("Initializing the Experiment Manager.")

        self.experiments = experiments
        self.results = []

    def run_all_experiments(self):
        """
        Run all experiments and store the results.
        """
        logging.info("Running all experiments.")

        for experiment in self.experiments:
            runner = ExperimentRunner(experiment)
            results = runner.run_experiment()
            self.results.append(results)
            print(results)

    def save_results(self, file_path: str = "experiment_results.csv"):
        """
        Save the experiment results to a CSV file.

        Args:
            file_path (str, optional): The file path to save the results.
                Defaults to "experiment_results.csv".
        """
        save_data(pd.DataFrame(self.results), file_path)
