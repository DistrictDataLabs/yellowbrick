# yellowbrick.datasets.path
# Helper functions for looking up dataset paths.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Thu Jul 26 14:10:51 2018 -0400
#
# ID: path.py [] benjamin@bengfort.com $

"""
Helper functions for looking up dataset paths.
"""

##########################################################################
## Imports
##########################################################################

import os

# from .download import download_data
from yellowbrick.exceptions import DataError


##########################################################################
## Fixtures
##########################################################################

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


##########################################################################
## Helper Functions
##########################################################################

def get_data_home(path=None):
    """
    Return the path of the Yellowbrick data directory. This folder is used by
    dataset loaders to avoid downloading data several times.

    By default, this folder is colocated with the code in the install directory
    so that data shipped with the package can be easily located. Alternatively
    it can be set by the YELLOWBRICK_DATA environment variable, or
    programmatically by giving a folder path. Note that the '~' symbol is
    expanded to the user home directory, and environment variables are also
    expanded when resolving the path.
    """
    if path is None:
        path = os.environ.get('YELLOWBRICK_DATA', FIXTURES)

    path = os.path.expanduser(path)
    path = os.path.expandvars(path)

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def find_dataset_path(dataset, data_home=None, fname=None, ext=".csv"):
    """
    Looks up the path to the dataset specified in the data home directory,
    which is found using the ``get_data_home`` function. By default data home
    is colocated with the code, but can be modified with the YELLOWBRICK_DATA
    environment variable, or passing in a different directory.

    The file returned will be by default, the name of the dataset in CSV
    format. Other files and extensions can be passed in to locate other data
    types or auxilliary files.

    If the dataset is not found a ``DataError`` is raised.

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

    ext : str, default: ".csv"
        The extension of the data to look up in the dataset path, if the fname
        is specified then the ext parameter is ignored. If ext is None then
        the directory of the dataset will be returned.

    Returns
    -------
    path : str
        A path to the requested file, guaranteed to exist if an exception is
        not raised during processing of the request.
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
        raise DataError((
            "could not find dataset at {} - does it need to be downloaded?"
        ).format(path))

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
