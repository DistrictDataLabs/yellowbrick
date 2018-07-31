# tests.test_datasets.test_loaders
# Test the dataset loading utilities
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Tue Jul 31 15:34:56 2018 -0400
#
# ID: test_loaders.py [] benjamin@bengfort.com $

"""
Test the dataset loading utilities
"""

##########################################################################
## Imports
##########################################################################

import pytest

from yellowbrick.datasets import *
from yellowbrick.datasets.base import Dataset, Corpus
from yellowbrick.datasets.path import dataset_exists


##########################################################################
## Test Cases
##########################################################################

class TestDatasetLoaders(object):
    """
    Test the dataset loading functions
    """

    def test_load_concrete(self):
        """
        Test loading the concrete regression dataset
        """
        data = load_concrete()
        assert isinstance(data, Dataset)
        assert dataset_exists("concrete")
        assert len(data.contents()) == 3
        assert len(data.README) > 0
        assert len(data.meta) > 0

    def test_load_energy(self):
        """
        Test loading the energy multi regression dataset
        """
        data = load_energy()
        assert isinstance(data, Dataset)
        assert dataset_exists("energy")
        assert len(data.contents()) == 3
        assert len(data.README) > 0
        assert len(data.meta) > 0

    def test_load_credit(self):
        """
        Test loading the credit binary classification dataset
        """
        data = load_credit()
        assert isinstance(data, Dataset)
        assert dataset_exists("credit")
        assert len(data.contents()) == 3
        assert len(data.README) > 0
        assert len(data.meta) > 0

    def test_load_occupancy(self):
        """
        Test loading the occupancy binary classification dataset
        """
        data = load_occupancy()
        assert isinstance(data, Dataset)
        assert dataset_exists("occupancy")
        assert len(data.contents()) == 3
        assert len(data.README) > 0
        assert len(data.meta) > 0

    def test_load_mushroom(self):
        """
        Test loading the mushroom binary classification dataset
        """
        data = load_mushroom()
        assert isinstance(data, Dataset)
        assert dataset_exists("mushroom")
        assert len(data.contents()) == 3
        assert len(data.README) > 0

        # meta.json is broken
        #assert len(data.meta) > 0

    def test_load_hobbies(self):
        """
        Test loading the hobbies text corpus dataset
        """
        data = load_hobbies()
        assert isinstance(data, Corpus)
        assert dataset_exists("hobbies")
        assert len(data.contents()) == 6
        assert len(data.README) > 0
        assert data.meta is None

    @pytest.mark.xfail(reason="requires specialized dtype")
    def test_load_game(self):
        """
        Test loading the game multiclass classification dataset
        """
        data = load_game()
        assert isinstance(data, Dataset)
        assert dataset_exists("game")
        assert len(data.contents()) == 3
        assert len(data.README) > 0
        assert len(data.meta) > 0

    def test_load_bikeshare(self):
        """
        Test loading the bikeshare regression dataset
        """
        data = load_bikeshare()
        assert isinstance(data, Dataset)
        assert dataset_exists("bikeshare")
        assert len(data.contents()) == 3
        assert len(data.README) > 0

        # meta.json is broken
        #assert len(data.meta) > 0

    def test_load_spam(self):
        """
        Test loading the spam binary classification dataset
        """
        data = load_spam()
        assert isinstance(data, Dataset)
        assert dataset_exists("spam")
        assert len(data.contents()) == 3
        assert len(data.README) > 0
        assert len(data.meta) > 0
