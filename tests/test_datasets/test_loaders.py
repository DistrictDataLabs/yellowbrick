# tests.test_datasets.test_loaders
# Test the dataset loading utilities
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Tue Jul 31 15:34:56 2018 -0400
#
# ID: test_loaders.py [7082742] benjamin@bengfort.com $

"""
Test the dataset loading utilities
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np

from unittest.mock import patch

from yellowbrick.datasets.loaders import *
from yellowbrick.datasets.loaders import DATASETS
from yellowbrick.datasets.base import Dataset, Corpus
from yellowbrick.datasets.path import dataset_exists, dataset_archive
from yellowbrick.datasets.path import find_dataset_path
from yellowbrick.exceptions import DatasetsError

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Assertion Helpers
##########################################################################


def assert_valid_dataset(data, name):
    __tracebackhide__ = True
    assert isinstance(data, Dataset), "not a Dataset object"
    assert name in DATASETS, "dataset not in manifest"

    assert dataset_exists(name), "dataset directory does not exist"
    assert dataset_archive(
        name, DATASETS[name]["signature"]
    ), "dataset archive does not match signature"
    assert (
        find_dataset_path(name, ext=".csv.gz", raises=False) is not None
    ), "no .csv.tgz in dataset"
    assert (
        find_dataset_path(name, ext=".npz", raises=False) is not None
    ), "no .npz in dataset"

    n_files = len(data.contents())
    assert n_files == 4 or n_files == 5, "not enough files in dataset"
    assert len(data.README) > 0, "readme contains no data"
    assert len(data.meta) > 0, "metadata is empty"

    if n_files == 5:
        assert len(data.citation) > 0, "citation.bib is empty"

    assert "features" in data.meta, "no features in metadata"
    assert "target" in data.meta, "no target in metadata"


def assert_valid_corpus(corpus, name):
    __tracebackhide__ = True
    assert isinstance(corpus, Corpus), "not a Corpus object"
    assert name in DATASETS, "corpus not in manifest"

    assert dataset_exists(name), "corpus directory does not exist"
    assert dataset_archive(
        name, DATASETS[name]["signature"]
    ), "corpus archive does not match signature"

    n_contents = len(corpus.contents())
    assert n_contents > 2, "not enough files/directories in corpus"
    assert len(corpus.README) > 0, "readme contains no data"
    assert corpus.citation is None or len(corpus.citation) > 0, "citation.bib is empty"


def assert_valid_pandas(data):
    __tracebackhide__ = True
    # Get raw data frame
    df = data.to_dataframe()
    assert isinstance(df, pd.DataFrame), "raw dataframe is wrong type"

    # Get pandas data
    X, y = data.to_pandas()
    assert isinstance(X, pd.DataFrame), "X is not a DataFrame"
    assert isinstance(y, pd.Series), "y is not a Series"

    # Assert pandas is returned from to_data()
    X, y = data.to_data()
    assert isinstance(X, pd.DataFrame), "to_data does not return pandas"
    assert isinstance(y, pd.Series), "to_data does not return pandas"


def assert_valid_numpy(data):
    __tracebackhide__ = True
    X, y = data.to_numpy()
    assert isinstance(X, np.ndarray), "X is not a numpy array"
    assert isinstance(y, np.ndarray), "y is not a numpy array"
    assert X.ndim == 2 and y.ndim == 1, "X and y dimensions are incorrect"

    # Patch pandas and make defaults assertions
    X, y = data.to_data()
    assert isinstance(X, np.ndarray), "to_data does not return numpy"
    assert isinstance(y, np.ndarray), "to_data does not return numpy"

    with pytest.raises(DatasetsError):
        data.to_pandas(), "exception not raised when pandas unavailable"


##########################################################################
## Test Cases
##########################################################################


class TestDatasetLoaders(object):
    """
    Test the dataset loading functions

    Broadly: test each of the dataset loaders to ensure that they are valid
    for their particular type of dataset and that they return X and y by
    default. Then test their shape to ensure that the dataset hasn't changed
    unexpectedly between versions. See ``test_load_concrete`` for a sketch.

    Final tests with parametrize test all loaders against Base classes.
    Make sure you scroll to the bottom and implement your loader in the
    correct test batch!
    """

    def test_load_concrete(self):
        """
        Test loading the concrete regression dataset
        """
        # Load the type-specific dataset wrapper and validate it
        data = load_concrete(return_dataset=True)
        assert_valid_dataset(data, "concrete")

        # Ensure that the default returns X, y to match documentation
        # Check shape to ensure no unexpected dataset changes have occured
        # before we push something to PyPI!
        X, y = load_concrete()
        assert X.shape == (1030, 8)
        assert y.shape == (1030,)

    def test_load_energy(self):
        """
        Test loading the energy multi regression dataset
        """
        data = load_energy(return_dataset=True)
        assert_valid_dataset(data, "energy")

        X, y = load_energy()
        assert X.shape == (768, 8)
        assert y.shape == (768,)

    def test_load_credit(self):
        """
        Test loading the credit binary classification dataset
        """
        data = load_credit(return_dataset=True)
        assert_valid_dataset(data, "credit")

        X, y = load_credit()
        assert X.shape == (30000, 23)
        assert y.shape == (30000,)

    def test_load_occupancy(self):
        """
        Test loading the occupancy binary classification dataset
        """
        data = load_occupancy(return_dataset=True)
        assert_valid_dataset(data, "occupancy")

        X, y = load_occupancy()
        assert X.shape == (20560, 5)
        assert y.shape == (20560,)

    def test_load_mushroom(self):
        """
        Test loading the mushroom binary classification dataset
        """
        data = load_mushroom(return_dataset=True)
        assert_valid_dataset(data, "mushroom")

        X, y = load_mushroom()
        assert X.shape == (8123, 3)
        assert y.shape == (8123,)

    def test_load_hobbies(self):
        """
        Test loading the hobbies text corpus dataset
        """
        corpus = load_hobbies()
        assert_valid_corpus(corpus, "hobbies")

        assert len(corpus.labels) == 5
        assert len(corpus.files) == 448
        assert len(corpus.data) == 448
        assert len(corpus.target) == 448

    def test_load_game(self):
        """
        Test loading the game multiclass classification dataset
        """
        data = load_game(return_dataset=True)
        assert_valid_dataset(data, "game")

        X, y = load_game()
        assert X.shape == (67557, 42)
        assert y.shape == (67557,)

    def test_load_bikeshare(self):
        """
        Test loading the bikeshare regression dataset
        """
        data = load_bikeshare(return_dataset=True)
        assert_valid_dataset(data, "bikeshare")

        X, y = load_bikeshare()
        assert X.shape == (17379, 12)
        assert y.shape == (17379,)

    def test_load_spam(self):
        """
        Test loading the spam binary classification dataset
        """
        data = load_spam(return_dataset=True)
        assert_valid_dataset(data, "spam")

        X, y = load_spam()
        assert X.shape == (4600, 57)
        assert y.shape == (4600,)

    def test_load_walking(self):
        """
        Test loading the walking activity clustering dataset
        """
        data = load_walking(return_dataset=True)
        assert_valid_dataset(data, "walking")

        X, y = load_walking()
        assert X.shape == (149332, 4)
        assert y.shape == (149332,)

    def test_load_nfl(self):
        """
        Test loading the nfl clustering dataset
        """
        data = load_nfl(return_dataset=True)
        assert_valid_dataset(data, "nfl")

        X, y = load_nfl()
        assert X.shape == (494, 23)
        assert y.shape == (494,)

    @pytest.mark.skipif(pd is None, reason="pandas is required for this test")
    @pytest.mark.parametrize(
        "loader",
        [
            load_bikeshare,
            load_concrete,
            load_credit,
            load_energy,
            load_game,
            load_mushroom,
            load_occupancy,
            load_spam,
            load_walking,
            load_nfl,
        ],
        ids=lambda l: l.__name__,
    )
    def test_load_pandas(self, loader):
        """
        Test loading datasets as pandas objects
        """
        data = loader(return_dataset=True)
        assert_valid_pandas(data)

    @patch("yellowbrick.datasets.base.pd", None)
    @pytest.mark.parametrize(
        "loader",
        [
            load_bikeshare,
            load_concrete,
            load_credit,
            load_energy,
            load_game,
            load_mushroom,
            load_occupancy,
            load_spam,
            load_walking,
            load_nfl,
        ],
        ids=lambda l: l.__name__,
    )
    def test_load_numpy(self, loader):
        """
        Test loading datasets as numpy defaults
        """
        data = loader(return_dataset=True)
        assert_valid_numpy(data)
