# yellowbrick.datasets.loaders
# Dataset loading utilities and primary API to the datasets module.
#
# Author:  Benjamin Bengfort
# Created: Tue Jul 31 13:31:23 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: loaders.py [7082742] benjamin@bengfort.com $

"""
Dataset loading utilities and primary API to the datasets module.
"""

##########################################################################
## Imports
##########################################################################

import os
import json

from .base import Dataset, Corpus

__all__ = [
    "load_concrete",
    "load_energy",
    "load_credit",
    "load_occupancy",
    "load_mushroom",
    "load_hobbies",
    "load_game",
    "load_bikeshare",
    "load_spam",
    "load_walking",
    "load_nfl",
]


##########################################################################
## Links and SHA 256 signature of Yellowbrick hosted datasets
##########################################################################

MANIFEST = os.path.join(os.path.dirname(__file__), "manifest.json")
with open(MANIFEST, "r") as f:
    DATASETS = json.load(f)


##########################################################################
## Specific loading utilities
##########################################################################


def _load_dataset(name, data_home=None, return_dataset=False):
    """
    Load a dataset by name and return specified format.
    """
    info = DATASETS[name]
    data = Dataset(name, data_home=data_home, **info)
    if return_dataset:
        return data
    return data.to_data()


def _load_corpus(name, data_home=None):
    """
    Load a corpus object by name.
    """
    info = DATASETS[name]
    return Corpus(name, data_home=data_home, **info)


def load_concrete(data_home=None, return_dataset=False):
    """
    Loads the concrete multivariate dataset that is well suited to regression
    tasks. The dataset contains 1030 instances and 8 real valued attributes
    with a continuous target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the ``$YELLOWBRICK_DATA`` envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from ``$YELLOWBRICK_DATA`` or the default returned by ``get_data_home``.

    return_dataset : bool, default=False
        Return the raw dataset object instead of X and y numpy arrays to
        get access to alternative targets, extra features, content and meta.

    Returns
    -------
    X : array-like with shape (n_instances, n_features) if return_dataset=False
        A pandas DataFrame or numpy array describing the instance features.

    y : array-like with shape (n_instances,) if return_dataset=False
        A pandas Series or numpy array describing the target vector.

    dataset : Dataset instance if return_dataset=True
        The Yellowbrick Dataset object provides an interface to accessing the
        data in a variety of formats as well as associated metadata and content.
    """
    return _load_dataset("concrete", data_home, return_dataset)


def load_energy(data_home=None, return_dataset=False):
    """
    Loads the energy multivariate dataset that is well suited to multi-output
    regression and classification tasks. The dataset contains 768 instances and
    8 real valued attributes with two continous targets.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the ``$YELLOWBRICK_DATA`` envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from ``$YELLOWBRICK_DATA`` or the default returned by ``get_data_home``.

    return_dataset : bool, default=False
        Return the raw dataset object instead of X and y numpy arrays to
        get access to alternative targets, extra features, content and meta.

    Returns
    -------
    X : array-like with shape (n_instances, n_features) if return_dataset=False
        A pandas DataFrame or numpy array describing the instance features.

    y : array-like with shape (n_instances,) if return_dataset=False
        A pandas Series or numpy array describing the target vector.

    dataset : Dataset instance if return_dataset=True
        The Yellowbrick Dataset object provides an interface to accessing the
        data in a variety of formats as well as associated metadata and content.
    """
    return _load_dataset("energy", data_home, return_dataset)


def load_credit(data_home=None, return_dataset=False):
    """
    Loads the credit multivariate dataset that is well suited to binary
    classification tasks. The dataset contains 30000 instances and 23 integer
    and real value attributes with a discrete target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the ``$YELLOWBRICK_DATA`` envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from ``$YELLOWBRICK_DATA`` or the default returned by ``get_data_home``.

    return_dataset : bool, default=False
        Return the raw dataset object instead of X and y numpy arrays to
        get access to alternative targets, extra features, content and meta.

    Returns
    -------
    X : array-like with shape (n_instances, n_features) if return_dataset=False
        A pandas DataFrame or numpy array describing the instance features.

    y : array-like with shape (n_instances,) if return_dataset=False
        A pandas Series or numpy array describing the target vector.

    dataset : Dataset instance if return_dataset=True
        The Yellowbrick Dataset object provides an interface to accessing the
        data in a variety of formats as well as associated metadata and content.
    """
    return _load_dataset("credit", data_home, return_dataset)


def load_occupancy(data_home=None, return_dataset=False):
    """
    Loads the occupancy multivariate, time-series dataset that is well suited
    to binary classification tasks. The dataset contains 20560 instances with
    5 real valued attributes and a discrete target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the ``$YELLOWBRICK_DATA`` envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from ``$YELLOWBRICK_DATA`` or the default returned by ``get_data_home``.

    return_dataset : bool, default=False
        Return the raw dataset object instead of X and y numpy arrays to
        get access to alternative targets, extra features, content and meta.

    Returns
    -------
    X : array-like with shape (n_instances, n_features) if return_dataset=False
        A pandas DataFrame or numpy array describing the instance features.

    y : array-like with shape (n_instances,) if return_dataset=False
        A pandas Series or numpy array describing the target vector.

    dataset : Dataset instance if return_dataset=True
        The Yellowbrick Dataset object provides an interface to accessing the
        data in a variety of formats as well as associated metadata and content.
    """
    return _load_dataset("occupancy", data_home, return_dataset)


def load_mushroom(data_home=None, return_dataset=False):
    """
    Loads the mushroom multivariate dataset that is well suited to binary
    classification tasks. The dataset contains 8123 instances with 3
    categorical attributes and a discrete target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the ``$YELLOWBRICK_DATA`` envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from ``$YELLOWBRICK_DATA`` or the default returned by ``get_data_home``.

    return_dataset : bool, default=False
        Return the raw dataset object instead of X and y numpy arrays to
        get access to alternative targets, extra features, content and meta.

    Returns
    -------
    X : array-like with shape (n_instances, n_features) if return_dataset=False
        A pandas DataFrame or numpy array describing the instance features.

    y : array-like with shape (n_instances,) if return_dataset=False
        A pandas Series or numpy array describing the target vector.

    dataset : Dataset instance if return_dataset=True
        The Yellowbrick Dataset object provides an interface to accessing the
        data in a variety of formats as well as associated metadata and content.
    """
    return _load_dataset("mushroom", data_home, return_dataset)


def load_hobbies(data_home=None):
    """
    Loads the hobbies text corpus that is well suited to classification,
    clustering, and text analysis tasks. The dataset contains 448 documents in
    5 categories with 7420 paragraphs, 14251 sentences, 288520 words, and a
    vocabulary of 23738.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the ``$YELLOWBRICK_DATA`` envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from ``$YELLOWBRICK_DATA`` or the default returned by ``get_data_home``.

    Returns
    -------
    dataset : Corpus
        The Yellowbrick Corpus object provides an interface to accessing the
        text documents and metadata associated with the corpus.
    """
    return _load_corpus("hobbies", data_home)


def load_game(data_home=None, return_dataset=False):
    """
    Load the Connect-4 game multivariate and spatial dataset that is well
    suited to multiclass classification tasks. The dataset contains 67557
    instances with 42 categorical attributes and a discrete target.

    Note that the game data is stored with categorical features that need to
    be numerically encoded before use with scikit-learn estimators. We
    recommend the use of the ``sklearn.preprocessing.OneHotEncoder`` for this
    task and to develop a ``Pipeline`` using this dataset.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the ``$YELLOWBRICK_DATA`` envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from ``$YELLOWBRICK_DATA`` or the default returned by ``get_data_home``.

    return_dataset : bool, default=False
        Return the raw dataset object instead of X and y numpy arrays to
        get access to alternative targets, extra features, content and meta.

    Returns
    -------
    X : array-like with shape (n_instances, n_features) if return_dataset=False
        A pandas DataFrame or numpy array describing the instance features.

    y : array-like with shape (n_instances,) if return_dataset=False
        A pandas Series or numpy array describing the target vector.

    dataset : Dataset instance if return_dataset=True
        The Yellowbrick Dataset object provides an interface to accessing the
        data in a variety of formats as well as associated metadata and content.
    """
    return _load_dataset("game", data_home, return_dataset)


def load_bikeshare(data_home=None, return_dataset=False):
    """
    Loads the bike sharing univariate dataset that is well suited to regression
    tasks. The dataset contains 17379 instances with 12 integer and real valued
    attributes and a continuous target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the ``$YELLOWBRICK_DATA`` envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from ``$YELLOWBRICK_DATA`` or the default returned by ``get_data_home``.

    return_dataset : bool, default=False
        Return the raw dataset object instead of X and y numpy arrays to
        get access to alternative targets, extra features, content and meta.

    Returns
    -------
    X : array-like with shape (n_instances, n_features) if return_dataset=False
        A pandas DataFrame or numpy array describing the instance features.

    y : array-like with shape (n_instances,) if return_dataset=False
        A pandas Series or numpy array describing the target vector.

    dataset : Dataset instance if return_dataset=True
        The Yellowbrick Dataset object provides an interface to accessing the
        data in a variety of formats as well as associated metadata and content.
    """
    return _load_dataset("bikeshare", data_home, return_dataset)


def load_spam(data_home=None, return_dataset=False):
    """
    Loads the email spam dataset that is weill suited to binary classification
    and threshold tasks. The dataset contains 4600 instances with 57 integer and
    real valued attributes and a discrete target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the ``$YELLOWBRICK_DATA`` envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from ``$YELLOWBRICK_DATA`` or the default returned by ``get_data_home``.

    return_dataset : bool, default=False
        Return the raw dataset object instead of X and y numpy arrays to
        get access to alternative targets, extra features, content and meta.

    Returns
    -------
    X : array-like with shape (n_instances, n_features) if return_dataset=False
        A pandas DataFrame or numpy array describing the instance features.

    y : array-like with shape (n_instances,) if return_dataset=False
        A pandas Series or numpy array describing the target vector.

    dataset : Dataset instance if return_dataset=True
        The Yellowbrick Dataset object provides an interface to accessing the
        data in a variety of formats as well as associated metadata and content.
    """
    return _load_dataset("spam", data_home, return_dataset)


def load_walking(data_home=None, return_dataset=False):
    """
    Loads the walking activity dataset that is weill suited to clustering and
    multi-label classification tasks. The dataset contains multi-variate time
    series data with 149,332 real valued measurements across 22 unique walkers.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the ``$YELLOWBRICK_DATA`` envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from ``$YELLOWBRICK_DATA`` or the default returned by ``get_data_home``.

    return_dataset : bool, default=False
        Return the raw dataset object instead of X and y numpy arrays to
        get access to alternative targets, extra features, content and meta.

    Returns
    -------
    X : array-like with shape (n_instances, n_features) if return_dataset=False
        A pandas DataFrame or numpy array describing the instance features.

    y : array-like with shape (n_instances,) if return_dataset=False
        A pandas Series or numpy array describing the target vector.

    dataset : Dataset instance if return_dataset=True
        The Yellowbrick Dataset object provides an interface to accessing the
        data in a variety of formats as well as associated metadata and content.
    """
    return _load_dataset("walking", data_home, return_dataset)


def load_nfl(data_home=None, return_dataset=False):
    """
    Loads the football receivers dataset that is well suited to clustering
    tasks. The dataset contains 494 instances with 28 integer, real valued, and
    categorical attributes and a discrete target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the ``$YELLOWBRICK_DATA`` envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from ``$YELLOWBRICK_DATA`` or the default returned by ``get_data_home``.

    return_dataset : bool, default=False
        Return the raw dataset object instead of X and y numpy arrays to
        get access to alternative targets, extra features, content and meta.

    Returns
    -------
    X : array-like with shape (n_instances, n_features) if return_dataset=False
        A pandas DataFrame or numpy array describing the instance features.

    y : array-like with shape (n_instances,) if return_dataset=False
        A pandas Series or numpy array describing the target vector.

    dataset : Dataset instance if return_dataset=True
        The Yellowbrick Dataset object provides an interface to accessing the
        data in a variety of formats as well as associated metadata and content.
    """
    return _load_dataset("nfl", data_home, return_dataset)
