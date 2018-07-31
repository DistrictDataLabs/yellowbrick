# yellowbrick.datasets.base
# Loading utilities for the yellowbrick datasets.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Author:   Raul Peralta <raulpl25@gmail.com>
# Created: Thu Jul 26 13:53:01 2018 -0400
#
# ID: base.py [] benjamin@bengfort.com $

"""
Loading utilities for the yellowbrick datasets.
"""

##########################################################################
## Imports
##########################################################################

import os
import json
import numpy as np

from .download import download_data
from .path import find_dataset_path, dataset_exists

from yellowbrick.utils.decorators import memoized

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Dataset Object
##########################################################################

class BaseDataset(object):
    """
    Base functionality for Dataset and Corpus objects.
    """

    def __init__(self, name, url=None, signature=None, data_home=None):
        self.name = name
        self.data_home = data_home
        self.url = url
        self.signature = signature

        # Check if the dataset exists, and if not - download it!
        if not dataset_exists(self.name, data_home=data_home):
            self.download()

    def download(self, replace=False):
        """
        Download the dataset from the hosted Yellowbrick data store and save
        it to the location specified by ``get_data_home``. The downloader
        verifies the download completed successfully and safely by comparing
        the expected signature with the SHA 256 signature of the downloaded
        archive file.

        Parameters
        ----------
        replace : bool, default: False
            If the data archive already exists, replace the dataset. If this is
            False and the dataset exists, an exception is raised.
        """
        download_data(
            self.url, self.signature, data_home=self.data_home,
            replace=replace, extract=True
        )

    def contents(self):
        """
        Contents returns a list of the files in the data directory.
        """
        data = find_dataset_path(
            self.name, data_home=self.data_home, ext=None
        )
        return os.listdir(data)

    @memoized
    def README(self):
        """
        Returns the contents of the README.md file that describes the dataset
        in detail and contains attribution information.
        """
        path = find_dataset_path(
            self.name, data_home=self.data_home, fname="README.md"
        )
        with open(path, 'r') as f:
            return f.read()

    @memoized
    def meta(self):
        """
        Returns the contents of the meta.json file that describes important
        attributes about the dataset and modifies the behavior of the loader.
        """
        path = find_dataset_path(
            self.name, data_home=self.data_home, fname="meta.json", raises=False
        )
        if path is None:
            return None

        with open(path, 'r') as f:
            return json.load(f)


class Dataset(BaseDataset):
    """
    Datasets contain a reference to data on disk and provide utilities for
    quickly loading files and objects into a variety of formats. The most
    common use of the Dataset object is to load example datasets provided by
    Yellowbrick to run the examples in the documentation.

    The dataset by default will return the data as a numpy array, however if
    Pandas is installed, it is possible to access the data as a DataFrame and
    Series object. In either case, the data is represented by a features table,
    X and a target vector, y.

    Parameters
    ----------
    name : str
        The name of the dataset; should either be a folder in data home or
        specified in the yellowbrick.datasets.DATASETS variable. This name is
        used to perform all lookups and identify the dataset externally.

    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    url : str, optional
        The web location where the archive file of the dataset can be
        downloaded from.

    signature : str, optional
        The signature of the data archive file, used to verify that the latest
        version of the data has been downloaded and that the download hasn't
        been corrupted or modified in anyway.
    """

    def as_matrix(self):
        """
        Returns the dataset as two numpy arrays: X and y.

        Returns
        -------
        X : array-like with shape (n_instances, n_features)
            A numpy array describing the instance features.

        y : array-like with shape (n_instances,)
            A numpy array describing the target vector.
        """
        path = find_dataset_path(self.name, data_home=self.data_home)
        return np.genfromtxt(path, dtype=float, delimiter=',', names=True)

    def as_pandas(self):
        """
        Returns the dataset as two pandas objects: X and y.

        Returns
        -------
        X : DataFrame with shape (n_instances, n_features)
            A pandas DataFrame containing feature data and named columns.

        y : Series with shape (n_instances,)
            A pandas Series containing target data and an index that matches
            the feature DataFrame index.
        """
        if pd is None:
            raise ImportError(
                "pandas is required to load DataFrame, it can be installed with pip"
            )

        path = find_dataset_path(self.name, data_home=self.data_home)
        return pd.read_csv(path)


class Corpus(BaseDataset):
    """
    Corpus datasets contain a reference to documents on disk and provide
    utilities for quickly loading text data for use in machine learning
    workflows. The most common use of the corpus is to load the text analysis
    examples from the Yellowbrick documentation.

    Parameters
    ----------
    name : str
        The name of the corpus; should either be a folder in data home or
        specified in the yellowbrick.datasets.DATASETS variable. This name is
        used to perform all lookups and identify the corpus externally.

    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    url : str, optional
        The web location where the archive file of the corpus can be
        downloaded from.

    signature : str, optional
        The signature of the data archive file, used to verify that the latest
        version of the data has been downloaded and that the download hasn't
        been corrupted or modified in anyway.
    """

    @memoized
    def path(self):
        return find_dataset_path(self.name, data_home=self.data_home, ext=None)

    @memoized
    def categories(self):
        return [
            cat for cat in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, cat))
        ]

    @property
    def files(self):
        return [
            name
            for cat in self.categories
            for name in os.listdir(os.path.join(self.path, cat))
        ]

    @property
    def data(self):
        def read(path):
            with open(path, 'r', encoding='UTF-8') as f:
                return f.read()

        return [
            read(f) for f in self.files
        ]

    @property
    def target(self):
        return [
            os.path.basename(os.path.dirname(f)) for f in self.files
        ]
