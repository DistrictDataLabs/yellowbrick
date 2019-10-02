# tests.test_datasets.test_paths
# Tests for the dataset path utilities
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Thu Jul 26 14:28:14 2018 -0400
#
# ID: test_path.py [7082742] benjamin@bengfort.com $

"""
Tests for the dataset path utilities
"""

##########################################################################
## Imports
##########################################################################

import os
import pytest
import contextlib

from yellowbrick.datasets.path import *
from yellowbrick.exceptions import DatasetsError


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


def test_find_dataset_path(tmpdir):
    """
    Test find_dataset_path with a specified data_home
    """

    # Create the dataset
    data_home = tmpdir.mkdir("fixtures")
    foo = data_home.mkdir("foo")

    # Test the default lookup of foo/foo.csv.gz
    fpath = foo.join("foo.csv.gz")
    fpath.write("1,2,3")
    assert find_dataset_path("foo", data_home=data_home) == fpath

    # Test the extension based lookup of foo/foo.npz
    fpath = foo.join("foo.npz")
    fpath.write("1234")
    assert find_dataset_path("foo", data_home=data_home, ext=".npz") == fpath

    # Test the fname based lookup of foo/data.txt
    fpath = foo.join("data.txt")
    fpath.write("there is data in this file")
    assert find_dataset_path("foo", data_home=data_home, fname="data.txt") == fpath


def test_missing_find_dataset_path(tmpdir):
    """
    Test find_dataset_path when the dataset does not exist
    """
    data_home = tmpdir.mkdir("fixtures")

    # When the data directory doesn't exist
    with pytest.raises(DatasetsError):
        find_dataset_path("foo", data_home=str(data_home))

    # When the data directory exists but no file is in the directory
    foo = data_home.mkdir("foo")
    with pytest.raises(DatasetsError):
        find_dataset_path("foo", data_home=str(data_home))

    # When the specified file doesn't exist
    fpath = foo.join("foo.csv")
    fpath.write("1,2,3")
    with pytest.raises(DatasetsError):
        find_dataset_path("foo", data_home=str(data_home), ext=".npz")


def test_suppress_exception_find_dataset_path(tmpdir):
    """
    Assert that find_dataset_path can suppress exceptions
    """
    data_home = str(tmpdir.mkdir("fixtures"))
    assert find_dataset_path("foo", data_home=data_home, raises=False) is None


def test_dataset_exists(tmpdir):
    """
    Test the dataset_exists helper function
    """
    data_home = tmpdir.mkdir("fixtures")
    assert not os.path.exists(str(data_home.join("foo")))

    # Test when directory doesn't exist
    assert not dataset_exists("foo", str(data_home))

    # Test when path exists but is file
    fpath = data_home.join("foo.txt")
    fpath.write("foo")

    assert not dataset_exists("foo.txt", str(data_home))

    # Test correct case
    data_home.mkdir("foo")
    assert dataset_exists("foo", str(data_home))


def test_dataset_archive(tmpdir):
    """
    Test the dataset_archive determines if an archive is up to date
    """
    sig = "49b3fc3143d727d7819fabd4365d7e7b29794089dc9fa1e5e452aeb0b33d5eda"
    data_home = tmpdir.mkdir("fixtures")

    # When archive does not exist
    assert not dataset_archive("foo", sig, data_home=str(data_home))

    # Create archive
    fpath = data_home.join("foo.zip")
    fpath.write("this is a data archive")

    # When archive exists
    assert dataset_archive("foo", sig, data_home=data_home)

    # When archive does not match signature
    assert not dataset_archive("foo", "abcd", data_home=data_home)


def test_cleanup_dataset(tmpdir):
    """
    Test cleanup_dataset removes both data dir and archive
    """
    data_home = tmpdir.mkdir("fixtures")

    # Make dataset and archive
    foo = data_home.mkdir("foo")
    fdata = foo.join("foo.csv")
    fdata.write("testing 1,2,3")

    fzip = data_home.join("foo.zip")
    fzip.write("this is the archive file")

    # Make sure the files exist
    assert os.path.exists(fzip)
    assert os.path.exists(fdata)

    # Cleanup the dataset
    cleanup_dataset("foo", data_home=data_home)

    # Files should be gone
    assert not os.path.exists(fzip)
    assert not os.path.exists(fdata)


def test_cleanup_dataset_no_data(tmpdir):
    """
    Assert cleanup_dataset fails gracefully if data and archive don't exist.
    """
    data_home = tmpdir.mkdir("fixtures")
    cleanup_dataset("foo", data_home=str(data_home))

    # Files should be gone
    assert not os.path.exists(str(data_home.join("foo.zip")))
    assert not os.path.exists(str(data_home.join("foo")))
