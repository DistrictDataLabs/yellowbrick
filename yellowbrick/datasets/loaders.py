# yellowbrick.datasets.loaders
# Dataset loading utilities and primary API to the datasets module.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Tue Jul 31 13:31:23 2018 -0400
#
# ID: loaders.py [] benjamin@bengfort.com $

"""
Dataset loading utilities and primary API to the datasets module.
"""

##########################################################################
## Imports
##########################################################################

from .base import Dataset, Corpus


##########################################################################
## Links and SHA 256 signature of Yellowbrick hosted datasets
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
## Specific loading utilities
##########################################################################

def load_concrete(data_home=None):
    """
    Loads the concrete multivariate dataset that is well suited to regression
    tasks. The dataset contains 1030 instances and 9 real valued attributes
    with a continuous target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the $YELLOWBRICK_DATA envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    Returns
    -------
    dataset : Dataset
        The Yellowbrick Dataset object provides an interface to accessing the
        data and metadata associated with the dataset.
    """
    name = 'concrete'
    info = DATASETS[name]
    return Dataset(name, data_home=data_home, **info)


def load_energy(data_home=None):
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
    with the ``data_home`` parameter or the $YELLOWBRICK_DATA envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    Returns
    -------
    dataset : Dataset
        The Yellowbrick Dataset object provides an interface to accessing the
        data and metadata associated with the dataset.
    """
    # name of the dataset
    name = 'energy'
    info = DATASETS[name]
    return Dataset(name, data_home=data_home, **info)


def load_credit(data_home=None):
    """
    Loads the credit multivariate dataset that is well suited to binary
    classification tasks. The dataset contains 30000 instances and 24 integer
    and real value attributes with a discrete target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the $YELLOWBRICK_DATA envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    Returns
    -------
    dataset : Dataset
        The Yellowbrick Dataset object provides an interface to accessing the
        data and metadata associated with the dataset.
    """
    # name of the dataset
    name = 'credit'
    info = DATASETS[name]
    return Dataset(name, data_home=data_home, **info)


def load_occupancy(data_home=None):
    """
    Loads the occupancy multivariate, time-series dataset that is well suited
    to binary classification tasks. The dataset contains 20560 instances with
    7 real valued attributes and a discrete target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the $YELLOWBRICK_DATA envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    Returns
    -------
    dataset : Dataset
        The Yellowbrick Dataset object provides an interface to accessing the
        data and metadata associated with the dataset.
    """
    # name of the dataset
    name = 'occupancy'
    info = DATASETS[name]
    return Dataset(name, data_home=data_home, **info)


def load_mushroom(data_home=None):
    """
    Loads the mushroom multivariate dataset that is well suited to binary
    classification tasks. The dataset contains 8124 instances with 4
    categorical attributes and a discrete target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the $YELLOWBRICK_DATA envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    Returns
    -------
    dataset : Dataset
        The Yellowbrick Dataset object provides an interface to accessing the
        data and metadata associated with the dataset.
    """
    # name of the dataset
    name = 'mushroom'
    info = DATASETS[name]
    return Dataset(name, data_home=data_home, **info)


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
    with the ``data_home`` parameter or the $YELLOWBRICK_DATA envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    Returns
    -------
    dataset : Corpus
        The Yellowbrick Corpus object provides an interface to accessing the
        text documents and metadata associated with the corpus.
    """
    # name of the dataset
    name = 'hobbies'
    info = DATASETS[name]
    return Corpus(name, data_home=data_home, **info)


def load_game(data_home=None):
    """
    Load the Connect-4 game multivariate and spatial dataset that is well
    suited to multiclass classification tasks. The dataset contains 67557
    instances with 42 categorical attributes and a discrete target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the $YELLOWBRICK_DATA envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    Returns
    -------
    dataset : Dataset
        The Yellowbrick Dataset object provides an interface to accessing the
        data and metadata associated with the dataset.
    """
    raise NotImplementedError("requires specialized datatype")
    # name of the dataset
    # name = 'game'
    # path = find_dataset_path(name, data_home=data_home)
    # dtype = np.array(['S1']*42+['|S4'])
    # return np.genfromtxt(path, dtype=dtype, delimiter=',', names=True)


def load_bikeshare(data_home=None):
    """
    Loads the bike sharing univariate dataset that is well suited to regression
    tasks. The dataset contains 17379 instances with 16 integer and real valued
    attributes and a continuous target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the $YELLOWBRICK_DATA envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    Returns
    -------
    dataset : Dataset
        The Yellowbrick Dataset object provides an interface to accessing the
        data and metadata associated with the dataset.
    """
    # name of the dataset
    name = 'bikeshare'
    info = DATASETS[name]
    return Dataset(name, data_home=data_home, **info)


def load_spam(data_home=None):
    """
    Loads the email spam dataset that is weill suited to binary classification
    and theshold tasks. The dataset contains 4601 instances with 57 integer and
    real valued attributes and a discrete target.

    The Yellowbrick datasets are hosted online and when requested, the dataset
    is downloaded to your local computer for use. Note that if the dataset
    hasn't been downloaded before, an Internet connection is required. However,
    if the data is cached locally, no data will be downloaded. Yellowbrick
    checks the known signature of the dataset with the data downloaded to
    ensure the download completes successfully.

    Datasets are stored alongside the code, but the location can be specified
    with the ``data_home`` parameter or the $YELLOWBRICK_DATA envvar.

    Parameters
    ----------
    data_home : str, optional
        The path on disk where data is stored. If not passed in, it is looked
        up from YELLOWBRICK_DATA or the default returned by ``get_data_home``.

    Returns
    -------
    dataset : Dataset
        The Yellowbrick Dataset object provides an interface to accessing the
        data and metadata associated with the dataset.
    """
    # name of the dataset
    name = 'spam'
    info = DATASETS[name]
    return Dataset(name, data_home=data_home, **info)
