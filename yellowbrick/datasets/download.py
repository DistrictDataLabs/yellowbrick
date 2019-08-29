# yellowbrick.datasets.download
# Downloads the example datasets for running the examples.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Author:   Raul Peralta
# Created:  Wed May 18 11:54:45 2016 -0400
#
# Copyright (C) 2018 The scikit-yb developers
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
import zipfile

from urllib.request import urlopen

from .signature import sha256sum
from .path import get_data_home, cleanup_dataset

from yellowbrick.exceptions import DatasetsError


# Downlod chunk size
CHUNK = 524288


##########################################################################
## Download functions
##########################################################################


def download_data(url, signature, data_home=None, replace=False, extract=True):
    """
    Downloads the zipped data set specified at the given URL, saving it to
    the data directory specified by ``get_data_home``. This function verifies
    the download with the given signature and extracts the archive.

    Parameters
    ----------
    url : str
        The URL of the dataset on the Internet to GET

    signature : str
        The SHA 256 hash of the dataset archive being downloaded to verify
        that the dataset has been correctly downloaded

    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    replace : bool, default: False
        If the data archive already exists, replace the dataset. If this is
        False and the dataset exists, an exception is raised.

    extract : bool, default: True
        Extract the archive file after downloading it
    """
    data_home = get_data_home(data_home)

    # Get the name of the file from the URL
    basename = os.path.basename(url)
    name, _ = os.path.splitext(basename)

    # Get the archive and data directory paths
    archive = os.path.join(data_home, basename)
    datadir = os.path.join(data_home, name)

    # If the archive exists cleanup or raise override exception
    if os.path.exists(archive):
        if not replace:
            raise DatasetsError(
                ("dataset already exists at {}, set replace=False to overwrite").format(
                    archive
                )
            )

        cleanup_dataset(name, data_home=data_home)

    # Create the output directory if it does not exist
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    # Fetch the response in a streaming fashion and write it to disk.
    response = urlopen(url)

    with open(archive, "wb") as f:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)

    # Compare the signature of the archive to the expected one
    if sha256sum(archive) != signature:
        raise ValueError("Download signature does not match hardcoded signature!")

    # If extract, extract the zipfile.
    if extract:
        zf = zipfile.ZipFile(archive)
        zf.extractall(path=data_home)
