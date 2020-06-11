# tests.conftest
# Global definitions for Yellowbrick pytest
#
# Author:  Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created: Fri Mar 02 11:53:55 2018 -0500
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: conftest.py [957cd53] benjamin@bengfort.com $

"""
Global definitions for Yellowbrick PyTest
"""

##########################################################################
## Imports
##########################################################################

import os
import matplotlib as mpl

from pytest_flakes import FlakesItem


##########################################################################
## Configure tests
##########################################################################


def pytest_configure(config):
    """
    This function is called by pytest for every plugin and conftest file
    after the command line arguments have been passed but before the
    session object is created and all of the tests are created. It is used
    to set a global configuration before all tests are run.

    Yellowbrick uses this function primarily to ensure that the matplotlib
    environment is setup correctly for all tests.
    """
    # This is redundant with the line in tests/__init__.py but ensures that
    # the backend is correctly set across all tests and plugins.
    mpl.use("Agg")

    # Travis-CI does not have san-serif so ensure standard fonts are used.
    # TODO: this is currently being reset before each test; needs fixing.
    mpl.rcParams["font.family"] = "DejaVu Sans"

##########################################################################
## PyTest Hooks
##########################################################################


def docline(obj):
    """
    Returns the first line of the object's docstring or None if
    there is no __doc__ on the object.
    """
    if not obj.__doc__:
        return None
    lines = list(filter(None, obj.__doc__.split("\n")))
    return lines[0].strip()


def pytest_itemcollected(item):
    """
    A reporting hook that is called when a test item is collected.

    This function can do many things to modify test output and test
    handling and in the future can be modified with those tasks. Right now
    we're currently only using this function to create node ids that
    pytest-spec knows how to parse and display as spec-style output.

    .. seealso:: https://stackoverflow.com/questions/28898919/use-docstrings-to-list-tests-in-py-test
    """

    # Ignore Session and PyFlake tests that are generated automatically
    if not hasattr(item.parent, "obj") or isinstance(item, FlakesItem):
        return

    # Collect test objects to inspect
    parent = item.parent.obj
    node = item.obj

    # Goal: produce a parsable string of the relative path, parent docstring
    # or class name, and the docstring of the test case, then set the nodeid
    # so that pytest-spec will correctly parse the information.
    path = os.path.relpath(str(item.fspath))
    prefix = docline(parent) or getattr(parent, "__name__", parent.__class__.__name__)
    suffix = docline(node) or node.__name__

    # Add parametrize or test generation id to distinguish it in output
    # TODO: this is broken with pytest 4.2 (no attribute _genid)
    if hasattr(item, "_genid") and item._genid:
        suffix += " ({})".format(item._genid)

    if prefix or suffix:
        item._nodeid = "::".join((path, prefix.strip(), suffix.strip()))
