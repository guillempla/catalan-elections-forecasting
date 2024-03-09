"""
Download the Censal Sections GIS data from the IEC website.

This module provides a class, DownloadCensalSectionsGisData,
that allows you to download the Censal Sections GIS data from the IEC website and save it.
"""

import os
import logging
from utils.rw_files import download_file, unzip_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# create a constant dictionary that stores a URL for each year
URLS = {
    "2022": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20220101_0.zip",
    "2021": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20210101_2.zip",
    "2020": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20200101_2.zip",
    "2019": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20190101_2.zip",
    "2018": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20180101_2.zip",
    "2017": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20170101_2.zip",
    "2016": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20160101_2.zip",
    "2015": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20150101_2.zip",
    "2014": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20140101_2.zip",
    "2013": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20130101_2.zip",
    "2012": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20120101_1.zip",
    "2011": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20110101_0.zip",
    "2010": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20100101_0.zip",
    "2009": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20090101_0.zip",
    "2008": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20080101_0.zip",
    "2007": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20070101_0.zip",
    "2006": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20060101_0.zip",
    "2005": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20050101_0.zip",
    "2004": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20040101_0.zip",
    "2003": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20030101_0.zip",
    "2002": "https://datacloud.icgc.cat/datacloud/bseccen_etrs89/shp/bseccenv10sh1f1_20020101_0.zip",
}


class DownloadCensalSectionsGisData:
    def __init__(
        self,
        year: str,
        output_path: str = "../data/raw/",
    ) -> None:
        """
        Initialize the DownloadCensalSectionsGisData object.

        Args:
            year (str): The year of the data to download.
            output_path (str): The path to save the data.
        """
        self.url = URLS[year]
        self.output_path = output_path

    def download(self) -> None:
        """
        Download the Censal Sections GIS data from the IEC website and save it.
        """
        logging.info("Downloading GIS data from %s.", self.url)

        # Determine the filename from the URL and set the download path
        filename = self.url.split("/")[-1]
        save_path = os.path.join(self.output_path, filename)
        print(save_path)

        # Download the file
        download_file(self.url, save_path)

        # TODO extract to a folder

        # Unzip the file
        unzip_file(save_path, self.output_path)  # Specify the extraction path

        # Optionally, remove the ZIP file after extraction
        os.remove(save_path)
        logging.info("Removed %s.", filename)
