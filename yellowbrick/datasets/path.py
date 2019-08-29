# yellowbrick.datasets.path
# Helper functions for looking up dataset paths.
#
# Author:  Benjamin Bengfort
# Created: Thu Jul 26 14:10:51 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: path.py [7082742] benjamin@bengfort.com $

"""
Helper functions for looking up dataset paths.
"""

##########################################################################
## Imports
##########################################################################

import os
import shutil

from .signature import sha256sum
from yellowbrick.exceptions import DatasetsError


##########################################################################
## Fixtures
##########################################################################

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


##########################################################################
## Dataset path utilities
##########################################################################


def get_data_home(path=None):
    """
    Return the path of the Yellowbrick data directory. This folder is used by
    dataset loaders to avoid downloading data several times.

    By default, this folder is colocated with the code in the install directory
    so that data shipped with the package can be easily located. Alternatively
    it can be set by the ``$YELLOWBRICK_DATA`` environment variable, or
    programmatically by giving a folder path. Note that the ``'~'`` symbol is
    expanded to the user home directory, and environment variables are also
    expanded when resolving the path.
    """
    if path is None:
        path = os.environ.get("YELLOWBRICK_DATA", FIXTURES)

    path = os.path.expanduser(path)
    path = os.path.expandvars(path)

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def find_dataset_path(dataset, data_home=None, fname=None, ext=".csv.gz", raises=True):
    """
    Looks up the path to the dataset specified in the data home directory,
    which is found using the ``get_data_home`` function. By default data home
    is colocated with the code, but can be modified with the YELLOWBRICK_DATA
    environment variable, or passing in a different directory.

    The file returned will be by default, the name of the dataset in compressed
    CSV format. Other files and extensions can be passed in to locate other data
    types or auxilliary files.

    If the dataset is not found a ``DatasetsError`` is raised by default.

    Parameters
    ----------
    dataset : str
        The name of the dataset; should either be a folder in data home or
        specified in the yellowbrick.datasets.DATASETS variable.

    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    fname : str, optional
        The filename to look up in the dataset path, by default it will be the
        name of the dataset. The fname must include an extension.

    ext : str, default: ".csv.gz"
        The extension of the data to look up in the dataset path, if the fname
        is specified then the ext parameter is ignored. If ext is None then
        the directory of the dataset will be returned.

    raises : bool, default: True
        If the path does not exist, raises a DatasetsError unless this flag is set
        to False, at which point None is returned (e.g. for checking if the
        path exists or not).

    Returns
    -------
    path : str or None
        A path to the requested file, guaranteed to exist if an exception is
        not raised during processing of the request (unless None is returned).

    raises : DatasetsError
        If raise is True and the path does not exist, raises a DatasetsError.
    """
    # Figure out the root directory of the datasets
    data_home = get_data_home(data_home)

    # Figure out the relative path to the dataset
    if fname is None:
        if ext is None:
            path = os.path.join(data_home, dataset)
        else:
            path = os.path.join(data_home, dataset, "{}{}".format(dataset, ext))
    else:
        path = os.path.join(data_home, dataset, fname)

    # Determine if the path exists
    if not os.path.exists(path):

        # Suppress exceptions if required
        if not raises:
            return None

        raise DatasetsError(
            ("could not find dataset at {} - does it need to be downloaded?").format(
                path
            )
        )

    return path


def dataset_exists(dataset, data_home=None):
    """
    Checks to see if a directory with the name of the specified dataset exists
    in the data home directory, found with ``get_data_home``.

    Parameters
    ----------
    dataset : str
        The name of the dataset; should either be a folder in data home or
        specified in the yellowbrick.datasets.DATASETS variable.

    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    Returns
    -------
    exists : bool
        If a folder with the dataset name is in the data home directory.
    """
    data_home = get_data_home(data_home)
    path = os.path.join(data_home, dataset)

    return os.path.exists(path) and os.path.isdir(path)


def dataset_archive(dataset, signature, data_home=None, ext=".zip"):
    """
    Checks to see if the dataset archive file exists in the data home directory,
    found with ``get_data_home``. By specifying the signature, this function
    also checks to see if the archive is the latest version by comparing the
    sha256sum of the local archive with the specified signature.

    Parameters
    ----------
    dataset : str
        The name of the dataset; should either be a folder in data home or
        specified in the yellowbrick.datasets.DATASETS variable.

    signature : str
        The SHA 256 signature of the dataset, used to determine if the archive
        is the latest version of the dataset or not.

    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    ext : str, default: ".zip"
        The extension of the archive file.

    Returns
    -------
    exists : bool
        True if the dataset archive exists and is the latest version.
    """
    data_home = get_data_home(data_home)
    path = os.path.join(data_home, dataset + ext)

    if os.path.exists(path) and os.path.isfile(path):
        return sha256sum(path) == signature

    return False


def cleanup_dataset(dataset, data_home=None, ext=".zip"):
    """
    Removes the dataset directory and archive file from the data home directory.

    Parameters
    ----------
    dataset : str
        The name of the dataset; should either be a folder in data home or
        specified in the yellowbrick.datasets.DATASETS variable.

    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    ext : str, default: ".zip"
        The extension of the archive file.

    Returns
    -------
    removed : int
        The number of objects removed from data_home.
    """
    removed = 0
    data_home = get_data_home(data_home)

    # Paths to remove
    datadir = os.path.join(data_home, dataset)
    archive = os.path.join(data_home, dataset + ext)

    # Remove directory and contents
    if os.path.exists(datadir):
        shutil.rmtree(datadir)
        removed += 1

    # Remove the archive file
    if os.path.exists(archive):
        os.remove(archive)
        removed += 1

    return removed
