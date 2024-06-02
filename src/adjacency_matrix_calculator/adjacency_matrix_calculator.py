"""
Module that calculates the adjacency matrix of census sections.
"""

import logging
import geopandas as gpd
import pandas as pd
import numpy as np

from utils.rw_files import load_data, save_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - Calculate Adjacency Matrix of Censal Sections - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class AdjacencyMatrixCalculator:
    """
    Class that calculates the adjacency matrix of census sections.
    """

    def __init__(self, censal_sections_path: str, output_path: str) -> None:
        """
        Initialize the AdjacencyMatrixCalculator.

        Args:
            censal_sections_path (str): The path to the censal sections data.
            output_path (str): The path to save the adjacency matrix.

        Returns:
            None
        """
        logging.info("Initializing AdjacencyMatrixCalculator.")

        self.censal_sections_gdf: gpd.GeoDataFrame = load_data(censal_sections_path)
        self.output_path: str = output_path

    def calculate_adjacency_matrix(self) -> None:
        """
        Calculate and save the adjacency matrix for a GeoDataFrame of census sections.

        Returns:
            None
        """
        logging.info("Calculating adjacency matrix.")

        # Extract the MUNDISSEC codes
        codes: np.ndarray = self.censal_sections_gdf["MUNDISSEC"].values
        n: int = len(codes)

        # Initialize the adjacency matrix with zeros
        adjacency_matrix: np.ndarray = np.zeros((n, n), dtype=int)

        # Create a spatial index for the geometries
        sindex = self.censal_sections_gdf.sindex

        for i in range(n):
            # Set the diagonal to -1
            adjacency_matrix[i, i] = -1

            # Get the potential neighbors using the spatial index
            possible_neighbors = list(
                sindex.intersection(self.censal_sections_gdf.geometry.iloc[i].bounds)
            )

            for j in possible_neighbors:
                if i != j and self.censal_sections_gdf.geometry.iloc[i].touches(
                    self.censal_sections_gdf.geometry.iloc[j]
                ):
                    adjacency_matrix[i, j] = 1

        # Convert the numpy array to a DataFrame for better readability
        adjacency_df: pd.DataFrame = pd.DataFrame(
            adjacency_matrix, index=codes, columns=codes
        )

        save_data(adjacency_df, self.output_path, save_csv=False, save_pickle=True)
