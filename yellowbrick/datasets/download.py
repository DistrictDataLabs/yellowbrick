# yellowbrick.datasets.download
# Downloads the example datasets for running the examples.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Author:   Raul Peralta <raulpl25@gmail.com>
# Created:  Wed May 18 11:54:45 2016 -0400
#
# ID: download.py [1f73d2b] benjamin@bengfort.com $

"""
Downloads the example datasets for running the examples.
"""

##########################################################################
## Imports
##########################################################################

import os
import six
import zipfile
import hashlib

if six.PY2:
    # backport for encoding in open for python2
    from io import open

try:
    from urllib.request import urlopen
except ImportError:
    # python 2
    from urllib2 import urlopen

from .base import DATASETS, FIXTURES

##########################################################################
## Functions
##########################################################################

def download(path=FIXTURES):
    """
    Downloads all the example datasets to the specified path.
    """
    for name, meta in DATASETS.items():
        download_data(name, data_dir=data_path)


##########################################################################
## Download functions
##########################################################################

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


##########################################################################
## Signature checking utility
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
