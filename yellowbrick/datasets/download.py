#!/usr/bin/env python
# download
# Downloads the example datasets for running the examples.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Author:   Raul Peralta <raulpl25@gmail.com>
# Created:  Wed May 18 11:54:45 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: download.py [1f73d2b] benjamin@bengfort.com $

"""
Downloads the example datasets for running the examples.
"""

##########################################################################
## Imports
##########################################################################

import os
import numpy as np

from .utils import load_numpy, load_corpus, download_data, DATASETS
from .utils import _lookup_path

##########################################################################
## Functions
##########################################################################

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def download_all(data_path=FIXTURES, verify=True):
    """
    Downloads all the example datasets. If verify is True then compare the
    download signature with the hardcoded signature. If extract is True then
    extract the contents of the zipfile to the given path.
    """
    for name, meta in DATASETS.items():
        download_data(name, data_dir=data_path)


def load_concrete(data_path=FIXTURES):
    """
    Downloads the 'concrete' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'concrete'
    data = load_numpy(name, data_path=data_path)
    return data


def load_energy(data_path=FIXTURES):
    """
    Downloads the 'energy' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'energy'
    data = load_numpy(name, data_path=data_path)
    return data


def load_credit(data_path=FIXTURES):
    """
    Downloads the 'credit' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'credit'
    data = load_numpy(name, data_path=data_path)
    return data


def load_occupancy(data_path=FIXTURES):
    """
    Downloads the 'occupancy' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'occupancy'
    data = load_numpy(name, data_path=data_path)
    return data


def load_mushroom(data_path=FIXTURES):
    """
    Downloads the 'mushroom' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'mushroom'
    data = load_numpy(name, data_path=data_path)
    return data


def load_hobbies(data_path=FIXTURES):
    """
    Downloads the 'hobbies' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'hobbies'
    data = load_corpus(name, data_path=data_path)
    return data


def load_game(data_path=FIXTURES):
    """
    Downloads the 'game' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'game'
    path = _lookup_path(name, data_path=data_path)
    dtype = np.array(['S1']*42+['|S4'])
    return np.genfromtxt(path, dtype=dtype, delimiter=',', names=True)


def load_bikeshare(data_path=FIXTURES):
    """
    Downloads the 'bikeshare' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'bikeshare'
    data = load_numpy(name, data_path=data_path)
    return data


def load_spam(data_path=FIXTURES):
    """
    Downloads the 'spam' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'spam'
    data = load_numpy(name, skip_header=True, data_path=data_path)
    return data
