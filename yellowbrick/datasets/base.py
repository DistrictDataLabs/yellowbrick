# yellowbrick.datasets.base
# Loading utilities for the yellowbrick datasets.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Author:   Raul Peralta
# Created: Thu Jul 26 13:53:01 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
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

from yellowbrick.exceptions import DatasetsError
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
        self.url = url
        self.name = name
        self.data_home = data_home
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
            self.url,
            self.signature,
            data_home=self.data_home,
            replace=replace,
            extract=True,
        )

    def contents(self):
        """
        Contents returns a list of the files in the data directory.
        """
        data = find_dataset_path(self.name, data_home=self.data_home, ext=None)
        return os.listdir(data)

    @memoized
    def README(self):
        """
        Returns the contents of the README.md file that describes the dataset
        in detail and contains attribution information.
        """
        path = find_dataset_path(self.name, data_home=self.data_home, fname="README.md")
        with open(path, "r") as f:
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

        with open(path, "r") as f:
            return json.load(f)

    @memoized
    def citation(self):
        """
        Returns the contents of the citation.bib file that describes the source
        and provenance of the dataset or to cite for academic work.
        """
        path = find_dataset_path(
            self.name, data_home=self.data_home, fname="meta.json", raises=False
        )
        if path is None:
            return None

        with open(path, "r") as f:
            return f.read()


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

    def to_data(self):
        """
        Returns the data contained in the dataset as X and y where X is the
        features matrix and y is the target vector. If pandas is installed,
        the data will be returned as DataFrame and Series objects. Otherwise,
        the data will be returned as two numpy arrays.

        Returns
        -------
        X : array-like with shape (n_instances, n_features)
            A pandas DataFrame or numpy array describing the instance features.

        y : array-like with shape (n_instances,)
            A pandas Series or numpy array describing the target vector.
        """
        if pd is not None:
            return self.to_pandas()
        return self.to_numpy()

    def to_numpy(self):
        """
        Returns the dataset as two numpy arrays: X and y.

        Returns
        -------
        X : array-like with shape (n_instances, n_features)
            A numpy array describing the instance features.

        y : array-like with shape (n_instances,)
            A numpy array describing the target vector.
        """
        path = find_dataset_path(self.name, ext=".npz", data_home=self.data_home)
        with np.load(path, allow_pickle=False) as npf:
            if "X" not in npf or "y" not in npf:
                raise DatasetsError(
                    (
                        "the downloaded dataset was improperly packaged without numpy "
                        "arrays - please report this bug to the Yellowbrick maintainers!"
                    )
                )

            # TODO: How to handle the case where y is None?
            return npf["X"], npf["y"]

    def to_pandas(self):
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
        # Ensure the metadata is valid before continuing
        if self.meta is None:
            raise DatasetsError(
                (
                    "the downloaded dataset was improperly packaged without meta.json "
                    "- please report this bug to the Yellowbrick maintainers!"
                )
            )

        if "features" not in self.meta or "target" not in self.meta:
            raise DatasetsError(
                (
                    "the downloaded dataset was improperly packaged without features "
                    "or target - please report this bug to the Yellowbrick maintainers!"
                )
            )

        # Load data frame and return features and target
        # TODO: Return y as None if there is no self.meta["target"]
        df = self.to_dataframe()
        return df[self.meta["features"]], df[self.meta["target"]]

    def to_dataframe(self):
        """
        Returns the entire dataset as a single pandas DataFrame.

        Returns
        -------
        df : DataFrame with shape (n_instances, n_columns)
            A pandas DataFrame containing the complete original data table
            including all targets (specified by the meta data) and all
            features (including those that might have been filtered out).
        """
        if pd is None:
            raise DatasetsError(
                "pandas is required to load DataFrame, it can be installed with pip"
            )

        path = find_dataset_path(self.name, ext=".csv.gz", data_home=self.data_home)
        return pd.read_csv(path, compression="gzip")


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
    def root(self):
        """
        Discovers and caches the root directory of the corpus.
        """
        return find_dataset_path(self.name, data_home=self.data_home, ext=None)

    @memoized
    def labels(self):
        """
        Return the unique labels assigned to the documents.
        """
        return [
            name
            for name in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, name))
        ]

    @property
    def files(self):
        """
        Returns the list of file names for all documents.
        """
        return [
            os.path.join(self.root, label, name)
            for label in self.labels
            for name in os.listdir(os.path.join(self.root, label))
        ]

    @property
    def data(self):
        """
        Read all of the documents from disk into an in-memory list.
        """

        def read(path):
            with open(path, "r", encoding="UTF-8") as f:
                return f.read()

        return [read(f) for f in self.files]

    @property
    def target(self):
        """
        Returns the label associated with each item in data.
        """
        return [os.path.basename(os.path.dirname(f)) for f in self.files]
