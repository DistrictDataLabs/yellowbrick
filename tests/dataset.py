# tests.dataset
# Helper functions for tests that utilize downloadable datasets.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Oct 13 19:55:53 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: dataset.py [8f4de77] benjamin@bengfort.com $

"""
Helper functions for tests that utilize downloadable datasets.
"""

##########################################################################
## Imports
##########################################################################

import os
import shutil
import hashlib
import zipfile
import numpy as np

from sklearn.datasets.base import Bunch

try:
    import requests
except ImportError:
    requests = None


##########################################################################
## Fixtures
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
    'hobbies': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/hobbies.zip',
        'signature': '415c8f68df1486d5d84a1d1757a5aa3035aef5ad63ede5013c261d622fbd29d8',
        'type': 'corpus',
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
}

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


##########################################################################
## Test Cases that Require Download
##########################################################################

class DatasetMixin(object):
    """
    Mixin for unittest.TestCase class to download datasets from S3 for
    testing real world machine learning visual diagnostics.
    """

    @staticmethod
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


    @staticmethod
    def download_data(url, path=FIXTURES, signature=None, extract=True):
        """
        Downloads the zipped data set specified at the given URL, saving it to
        the output path specified. This function verifies the download with the
        given signature (if supplied) and extracts the zip file if requested.
        """
        if requests is None:
            raise ImportError(
                "The requests module is required to download data --\n"
                "please install it with pip install requests."
            )

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
            dlsignature = DatasetMixin.sha256sum(dlpath)
            if signature != dlsignature:
                raise ValueError(
                    "Download signature does not match hardcoded signature!"
                )

        # If extract, extract the zipfile.
        if extract:
            zf = zipfile.ZipFile(dlpath)
            zf.extractall(path)


    @staticmethod
    def download_all(path=FIXTURES, verify=True, extract=True):
        """
        Downloads all the example datasets. If verify is True then compare the
        download signature with the hardcoded signature. If extract is True then
        extract the contents of the zipfile to the given path.
        """
        for name, meta in DATASETS.items():
            url = meta['url']
            signature = meta['signature'] if verify else None

            DatasetMixin.download_data(
                url, path=path, signature=signature, extract=extract
            )

    @staticmethod
    def remove_all(fixtures=FIXTURES):
        """
        Removes all the downloaded datasets as clean up
        """
        shutil.rmtree(fixtures)

    @staticmethod
    def load_data(name, fixtures=FIXTURES):
        """
        Loads the numpy matrix from the specified data set, downloads it if
        it hasn't already been downloaded.
        """
        # Just in case this is a corpus data set, then do that instead.
        if DATASETS[name]['type'] == 'corpus':
            return DatasetMixin.load_corpus(name, fixtures)

        path = os.path.join(fixtures, name, "{}.csv".format(name))
        if not os.path.exists(path):
            DatasetMixin.download_all(path=fixtures)

        return np.genfromtxt(path, dtype=float, delimiter=',', names=True)

    @staticmethod
    def load_corpus(name, fixtures=FIXTURES):
        """
        Loads a sklearn Bunch with the corpus and downloads it if it hasn't
        already been downloaded. Used to test text visualizers.
        """
        path = os.path.join(fixtures, name)
        if not os.path.exists(path):
            DatasetMixin.download_all(path=fixtures)

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

                with open(os.path.join(path, cat, name), 'r') as f:
                    data.append(f.read())

        # Return the data bunch for use similar to the newsgroups example
        return Bunch(
            categories=categories,
            files=files,
            data=data,
            target=target,
        )
