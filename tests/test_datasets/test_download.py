import unittest

import numpy as np
from sklearn.utils import Bunch

from yellowbrick.datasets import *


class TestDataDownloaders(unittest.TestCase):
    """
    Test the dataset loading functions
    """

    def test_load_concrete(self):
        data = load_concrete()
        self.assertIsInstance(data, np.ndarray)

    def test_load_energy(self):
        data = load_energy()
        self.assertIsInstance(data, np.ndarray)

    def test_load_occupancy(self):
        data = load_occupancy()
        self.assertIsInstance(data, np.ndarray)

    def test_load_mushroom(self):
        data = load_mushroom()
        self.assertIsInstance(data, np.ndarray)

    def test_load_hobbies(self):
        data = load_hobbies()
        self.assertIsInstance(data, Bunch)

    def test_load_game(self):
        data = load_game()
        self.assertIsInstance(data, np.ndarray)

    def test_load_bikeshare(self):
        data = load_bikeshare()
        self.assertIsInstance(data, np.ndarray)

    def test_load_spam(self):
        data = load_spam()
        self.assertIsInstance(data, np.ndarray)
