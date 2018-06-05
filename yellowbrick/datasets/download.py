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
import sys
import hashlib
import zipfile
import json
import csv
import numpy as np

try:
    import requests
except ImportError:
    print((
        "The requests module is required to download data --\n"
        "please install it with pip install requests."
    ))
    sys.exit(1)


##########################################################################
## Links and MD5 hash of datasets
##########################################################################

DATASETS = {
    'concrete': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/concrete.zip',
        'signature': 'b9ea5f26a7bb272a040e2f1a993b26babbf8dc4a04ab8198bb315ca66d71f10d',
    },
    'energy': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/energy.zip',
        'signature': '19fb86f3bcdde208eed46944172cb643ef6a7d58da103fb568fae43205ed89d3',
    },
    'credit': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/credit.zip',
        'signature': '4a91339c69f55e18f3f48004328fbcb7868070b618208fed099920427b084e5e',
    },
    'occupancy': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/occupancy.zip',
        'signature': '429cfe376dc9929a1fa528da89f0e1626e34e19695f3f555d8954025bbc522b8',
    },
    'mushroom': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/mushroom.zip',
        'signature': '884c43cb70db35d211c67b1cf6a3683b2b4569393d2789d5c07840da4dc85ba8',
    },
    'hobbies': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/hobbies.zip',
        'signature': '415c8f68df1486d5d84a1d1757a5aa3035aef5ad63ede5013c261d622fbd29d8',
    },
    'game': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/game.zip',
        'signature': 'b1bd85789a014a898daa34cb5f89ceab6d2cd6488a2e572187e34aa4ec21a43b',
    },
    'bikeshare': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/bikeshare.zip',
        'signature': 'a9b440f65549746dff680c92ff8bdca3c7265f09db1cf09e708e6e26fc8aba44',
    },
    'spam': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/spam.zip',
        'signature': '65be21196ba3d8448847409b70a67d761f873f30719c807600eb516d7aef1de1',
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


def download_data(url, path='data', signature=None, extract=True):
    """
    Downloads the zipped data set specified at the given URL, saving it to
    the output path specified. This function verifies the download with the
    given signature (if supplied) and extracts the zip file if requested.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(path):
        os.mkdir(path)

    # Get the name of the file from the URL
    name = os.path.basename(url)
    dlpath = os.path.join(path, name)

    # Fetch the response in a streaming fashion and write it to disk.
    response = requests.get(url, stream=True)
    with open(dlpath, 'wb') as f:
        for chunk in response.iter_content(65536):
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
        zf.extractall(path)


def download_all(path='data', verify=True, extract=True):
    """
    Downloads all the example datasets. If verify is True then compare the
    download signature with the hardcoded signature. If extract is True then
    extract the contents of the zipfile to the given path.
    """
    for name, meta in DATASETS.items():
        url = meta['url']
        signature = meta['signature'] if verify else None

        download_data(url, path=path, signature=signature, extract=extract)


def _load_file_data(name, path='data', extract=True):
    """
    Returns the information of the specified dataset.
    """
    url = DATASETS[name]['url']
    signature = DATASETS[name]['signature']
    download_data(url, path=path, signature=signature, extract=extract)
    with open(os.path.join(path, name, 'meta.json')) as meta_file:
        feature_names = json.load(meta_file)
    with open(os.path.join(path, name, 'README.md')) as readme_file:
        description = readme_file.read()
    with open(os.path.join(path, name, '{0}.csv'.format(name))) as csv_file:
        data_file = csv.reader(csv_file)
        # removing columns name
        next(data_file)
        data = np.asarray([line for line in data_file])
    result = {'data': data, 'DESCR': description}
    for k, v in feature_names.items():
        result[k] = v
    return result


def load_concrete(path='data', extract=True):
    """
    Downloads the 'concrete' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'concrete'
    data = _load_file_data(name, path, extract)
    return data


def load_energy(path='data', extract=True):
    """
    Downloads the 'energy' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'energy'
    data = _load_file_data(name, path, extract)
    return data


def load_credit(path='data', extract=True):
    """
    Downloads the 'credit' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'credit'
    data = _load_file_data(name, path, extract)
    return data


def load_occupancy(path='data', extract=True):
    """
    Downloads the 'occupancy' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'occupancy'
    data = _load_file_data(name, path, extract)
    return data


def load_mushroom(path='data', extract=True):
    """
    Downloads the 'mushroom' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'mushroom'
    data = _load_file_data(name, path, extract)
    return data


def load_hobbies(path='data', extract=True):
    """
    Downloads the 'hobbies' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'hobbies'
    data = _load_file_data(name, path, extract)
    return data


def load_game(path='data', extract=True):
    """
    Downloads the 'game' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'game'
    data = _load_file_data(name, path, extract)
    return data


def load_bikeshare(path='data', extract=True):
    """
    Downloads the 'bikeshare' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'bikeshare'
    data = _load_file_data(name, path, extract)
    return data


def load_spam(path='data', extract=True):
    """
    Downloads the 'spam' dataset, saving it to the output
    path specified and returns the data.
    """
    # name of the dataset
    name = 'spam'
    data = _load_file_data(name, path, extract)
    return data


if __name__ == '__main__':
    path = 'data'
    download_all(path)
    print("Downloaded datasets to {}".format(os.path.abspath(path)))
