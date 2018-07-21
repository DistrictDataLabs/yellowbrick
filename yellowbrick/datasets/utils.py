#!/usr/bin/env python
"""
Utils for downloading datasets for running the examples.
"""

##########################################################################
## Imports
##########################################################################

import os
import six
import hashlib
import zipfile
import numpy as np

from sklearn.datasets.base import Bunch

if six.PY2:
    # backport for encoding in open for python2
    from io import open

try:
    from urllib.request import urlopen
except ImportError:
    # python 2
    from urllib2 import urlopen

try:
    import pandas as pd
except ImportError:
    pd = None

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
## Download functions
##########################################################################

def sha256sum(path, blocksize=65536):
    """
    Computes the SHA256 signature of a file to verify that the file has not
    been modified in transit and that it is the correct version of the data.
    """
    sig = hashlib.sha256()
    with open(path, 'rb') as f:
        buf = f.read(blocksize)
        while len(buf) > 0:
            sig.update(buf)
            buf = f.read(blocksize)
    return sig.hexdigest()


def download_data(name, data_dir=None, signature=None, extract=True):
    """
    Downloads the zipped data set specified at the given URL, saving it to
    the output path specified. This function verifies the download with the
    given signature (if supplied) and extracts the zip file if requested.
    """

    # Create the fixture directory
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    dataset = DATASETS[name]
    url = dataset['url']

    # Get the name of the file from the URL
    filename = os.path.basename(url)
    dlpath = os.path.join(data_dir, filename)
    dataset_path = os.path.join(data_dir, name)

    #Create the output directory if it does not exist
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    # Fetch the response in a streaming fashion and write it to disk.
    response = urlopen(url)
    CHUNK = 16 * 1024
    with open(dlpath, 'wb') as f:

        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)

    # If verify, compare the signature
    if signature is not None:
        dlsignature = sha256sum(dlpath)
        if signature != dlsignature:
            raise ValueError(
                "Download signature does not match hardcoded signature!"
            )

    # If extract, extract the zipfile.
    if extract:
        zf = zipfile.ZipFile(dlpath)
        zf.extractall(path=data_dir)

def load_numpy(name, data_path=None, **kwargs):
    """
    Loads the numpy matrix from the specified data set, downloads it if
    it hasn't already been downloaded.
    """

    path = _lookup_path(name, data_path=data_path)
    return np.genfromtxt(path, dtype=float, delimiter=',', names=True, **kwargs)


def load_corpus(name, data_path=None):
    """
    Loads a sklearn Bunch with the corpus and downloads it if it hasn't
    already been downloaded. Used to test text visualizers.
    """
    path = _lookup_path(name, data_path=data_path, ext=None)

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

def _lookup_path(name, data_path=None, ext=".csv"):
    """
    Looks up the path to the dataset, downloading it if necessary
    """
    if ext is None:
        path = os.path.join(data_path, name)
    else:
        path = os.path.join(data_path, name, "{}{}".format(name, ext))

    if not os.path.exists(path):
        download_data(name, signature=None, extract=True, data_dir=data_path)

    return path
