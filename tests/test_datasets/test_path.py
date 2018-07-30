# tests.test_datasets.test_paths
# Tests for the dataset path utilities
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Thu Jul 26 14:28:14 2018 -0400
#
# ID: test_paths.py [] benjamin@bengfort.com $

"""
Tests for the dataset path utilities
"""

##########################################################################
## Imports
##########################################################################

import os
import contextlib

from yellowbrick.datasets.path import *


##########################################################################
## Utilities
##########################################################################

@contextlib.contextmanager
def environ(**env):
    """
    Temporarily set the environment variables for a test, restoring them when
    the test is complete (e.g. when the context manager exits).

    Parameters
    ----------
    env : dict
        The environment variables that should exist in context
    """
    old_env = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)



##########################################################################
## Test Cases
##########################################################################

def test_get_data_home_fixtures():
    """
    get_data_home should return fixtures by default
    """
    assert get_data_home() == FIXTURES
    assert os.path.exists(FIXTURES)


def test_get_data_home_env(tmpdir):
    """
    get_data_home should return the environment variable if set
    """
    path = str(tmpdir.mkdir("fixtures").join("foo"))
    assert not os.path.exists(path)

    with environ(YELLOWBRICK_DATA=path):
        assert get_data_home() == path
        assert os.path.exists(path)


def test_get_data_home_specified(tmpdir):
    """
    get_data_home should return a passed in path
    """
    path = str(tmpdir.mkdir("fixtures").join("foo"))
    assert not os.path.exists(path)

    assert get_data_home(path) == path
    assert os.path.exists(path)




def test_dataset_exists(tmpdir):
    """
    Test the dataset_exists helper function
    """
    data_home = tmpdir.mkdir("fixtures")
    assert not os.path.exists(data_home.join("foo"))

    # Test when directory doesn't exist
    assert not dataset_exists("foo", data_home)

    # Test when path exists but is file
    fpath = data_home.join("foo.txt")
    fpath.write("foo")

    assert not dataset_exists("foo.txt", data_home)

    # Test correct case
    data_home.mkdir("foo")
    assert dataset_exists("foo", data_home)
