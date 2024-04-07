"""
Transform censal sections data and results data into a single output dataframe
"""

from typing import List, Tuple

import logging
import pandas as pd
import numpy as np
import geopandas as gpd
from utils.rw_files import load_data, save_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class TransformData:
    """
    Transform censal sections data and results data into a single output dataframe
    """

    def __init__(self, censal_sections_path: str, results_path: str) -> None:
        """
        Initialize the class with the paths to the censal sections and results data
        """
        self.censal_sections_path = censal_sections_path
        self.results_path = results_path

    def transform_data(self) -> None:
        """
        Transform censal sections data and results data into a single output dataframe
        """
        # # Read censal sections data
        # censal_sections = load_data(self.censal_sections_path)

        # # Read results data
        # results = load_data(self.results_path)

        # # Merge censal sections and results data
        # output = pd.merge(censal_sections, results, on="section_id", how="inner")

        # save_data(output, self.results_path)
        pass
