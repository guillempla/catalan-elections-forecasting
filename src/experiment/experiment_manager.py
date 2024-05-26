"""
Module to execute multiple experiments and store the results.
"""

from typing import Dict, List
import pandas as pd

from experiment.experiment_runner import ExperimentRunner
from utils.rw_files import save_data


class ExperimentManager:
    """
    Class to manage and run experiments.

    Args:
        experiments (List[Dict]): A list of dictionaries representing the experiments to run.
    """

    def __init__(self, experiments: List[Dict]):
        self.experiments = experiments
        self.results = []

    def run_all_experiments(self):
        """
        Run all experiments and store the results.
        """
        for experiment in self.experiments:
            runner = ExperimentRunner(experiment)
            results = runner.run_experiment()
            print(results)

    def save_results(self, file_path: str = "experiment_results.csv"):
        """
        Save the experiment results to a CSV file.

        Args:
            file_path (str, optional): The file path to save the results.
                Defaults to "experiment_results.csv".
        """
        save_data(pd.DataFrame(self.results), file_path)
