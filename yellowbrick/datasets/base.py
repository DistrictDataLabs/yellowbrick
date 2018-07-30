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
import numpy as np

from sklearn.datasets.base import Bunch

from .path import find_dataset_path, FIXTURES


##########################################################################
## Links and MD5 hash of datasets
##########################################################################

DATASETS = {
    'concrete': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/concrete.zip',
        'signature': 'b9ea5f26a7bb272a040e2f1a993b26babbf8dc4a04ab8198bb315ca66d71f10d',
        'type': 'numpy',
    },
    'energy': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/energy.zip',
        'signature': '19fb86f3bcdde208eed46944172cb643ef6a7d58da103fb568fae43205ed89d3',
        'type': 'numpy',
    },
    'credit': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/credit.zip',
        'signature': '4a91339c69f55e18f3f48004328fbcb7868070b618208fed099920427b084e5e',
        'type': 'numpy',
    },
    'occupancy': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/occupancy.zip',
        'signature': '429cfe376dc9929a1fa528da89f0e1626e34e19695f3f555d8954025bbc522b8',
        'type': 'numpy',
    },
    'mushroom': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/mushroom.zip',
        'signature': '884c43cb70db35d211c67b1cf6a3683b2b4569393d2789d5c07840da4dc85ba8',
        'type': 'numpy',
    },
    'game': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/game.zip',
        'signature': 'b1bd85789a014a898daa34cb5f89ceab6d2cd6488a2e572187e34aa4ec21a43b',
        'type': 'numpy',
    },
    'bikeshare': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/bikeshare.zip',
        'signature': 'a9b440f65549746dff680c92ff8bdca3c7265f09db1cf09e708e6e26fc8aba44',
        'type': 'numpy',
    },
    'spam': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/spam.zip',
        'signature': '65be21196ba3d8448847409b70a67d761f873f30719c807600eb516d7aef1de1',
        'type': 'numpy',
    },
    'hobbies': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/hobbies.zip',
        'signature': '415c8f68df1486d5d84a1d1757a5aa3035aef5ad63ede5013c261d622fbd29d8',
        'type': 'corpus',
    },
}


##########################################################################
## General loading utilities
##########################################################################

def load_numpy(name, data_home=None, **kwargs):
    """
    Loads the numpy matrix from the specified data set, downloads it if
    it hasn't already been downloaded.
    """

    path = find_dataset_path(name, data_home=data_home)
    return np.genfromtxt(path, dtype=float, delimiter=',', names=True, **kwargs)


def load_corpus(name, data_home=None):
    """
    Loads a sklearn Bunch with the corpus and downloads it if it hasn't
    already been downloaded. Used to test text visualizers.
    """
    path = find_dataset_path(name, data_home=data_home, ext=None)

    # Read the directories in the directory as the categories.
    categories = [
        cat for cat in os.listdir(path)
        if os.path.isdir(os.path.join(path, cat))
    ]

    files  = [] # holds the file names relative to the root
    data   = [] # holds the text read from the file
    target = [] # holds the string of the category

    # Load the data from the files in the corpus
    for cat in categories:
        for name in os.listdir(os.path.join(path, cat)):
            files.append(os.path.join(path, cat, name))
            target.append(cat)

            with open(os.path.join(path, cat, name), 'r', encoding='UTF-8') as f:
                data.append(f.read())

    # Return the data bunch for use similar to the newsgroups example
    return Bunch(
        categories=categories,
        files=files,
        data=data,
        target=target,
    )


##########################################################################
## Specific loading utilities
##########################################################################

def load_concrete(data_home=FIXTURES):
    """
    Downloads the 'concrete' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'concrete'
    data = load_numpy(name, data_home=data_home)
    return data


def load_energy(data_home=FIXTURES):
    """
    Downloads the 'energy' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'energy'
    data = load_numpy(name, data_home=data_home)
    return data


def load_credit(data_home=FIXTURES):
    """
    Downloads the 'credit' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'credit'
    data = load_numpy(name, data_home=data_home)
    return data


def load_occupancy(data_home=FIXTURES):
    """
    Downloads the 'occupancy' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'occupancy'
    data = load_numpy(name, data_home=data_home)
    return data


def load_mushroom(data_home=FIXTURES):
    """
    Downloads the 'mushroom' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'mushroom'
    data = load_numpy(name, data_home=data_home)
    return data


def load_hobbies(data_home=FIXTURES):
    """
    Downloads the 'hobbies' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'hobbies'
    data = load_corpus(name, data_home=data_home)
    return data


def load_game(data_home=FIXTURES):
    """
    Downloads the 'game' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'game'
    path = find_dataset_path(name, data_home=data_home)
    dtype = np.array(['S1']*42+['|S4'])
    return np.genfromtxt(path, dtype=dtype, delimiter=',', names=True)


def load_bikeshare(data_home=FIXTURES):
    """
    Downloads the 'bikeshare' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'bikeshare'
    data = load_numpy(name, data_home=data_home)
    return data


def load_spam(data_home=FIXTURES):
    """
    Downloads the 'spam' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'spam'
    data = load_numpy(name, skip_header=True, data_home=data_home)
    return data
