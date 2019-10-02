# tests.test_datasets.test_download
# Tests the download from S3 to ensure data is accessible.
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Tue Jan 01 15:06:05 2019 -0500
#
# For license information, see LICENSE.txt
#
# ID: test_download.py [57aab02] ndanielsen@users.noreply.github.com $

"""
Tests the download from S3 to ensure data is accessible.
"""

##########################################################################
## Imports
##########################################################################

import pytest

from yellowbrick.datasets.loaders import *
from yellowbrick.datasets.loaders import DATASETS
from yellowbrick.datasets.path import dataset_exists, dataset_archive


@pytest.mark.parametrize(
    "loader",
    [
        load_bikeshare,
        load_concrete,
        load_credit,
        load_energy,
        load_game,
        load_mushroom,
        load_occupancy,
        load_spam,
        load_walking,
        load_hobbies,
        load_nfl,
    ],
    ids=lambda l: l.__name__,
)
def test_loader_download(tmpdir, loader):
    """
    Test download of dataset when it does not exist (requires Internet connection!)
    """
    name = loader.__name__[len("load_") :]
    data_home = str(tmpdir.mkdir("datasets"))

    # The dataset should not exist
    assert not dataset_exists(name, data_home=data_home)
    assert not dataset_archive(name, DATASETS[name]["signature"], data_home=data_home)

    # Load the dataset
    loader(data_home=data_home)

    # The dataset should have been downloaded
    assert dataset_exists(name, data_home=data_home)
    assert dataset_archive(name, DATASETS[name]["signature"], data_home=data_home)
