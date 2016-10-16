# tests.dataset
# Helper functions for tests that utilize downloadable datasets.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Oct 13 19:55:53 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: dataset.py [] benjamin@bengfort.com $

"""
Helper functions for tests that utilize downloadable datasets.
"""

##########################################################################
## Imports
##########################################################################

import os
import sys
import shutil
import hashlib
import zipfile
import numpy as np

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
    },
    'credit': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/credit.zip',
        'signature': '4a91339c69f55e18f3f48004328fbcb7868070b618208fed099920427b084e5e',
    },
    'occupancy': {
        'url': 'https://s3.amazonaws.com/ddl-data-lake/yellowbrick/occupancy.zip',
        'signature': '429cfe376dc9929a1fa528da89f0e1626e34e19695f3f555d8954025bbc522b8',
    }
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

        path = os.path.join(fixtures, name, "{}.csv".format(name))
        if not os.path.exists(path):
            DatasetMixin.download_all(path=fixtures)

        return np.genfromtxt(path, dtype=float, delimiter=',', names=True)
