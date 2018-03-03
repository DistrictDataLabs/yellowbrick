# tests.conftest
# Global definitions for Yellowbrick PyTest
#
# Author:  Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created: Fri Mar 02 11:53:55 2018 -0500
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: conftest.py [] benjamin@bengfort.com $

"""
Global definitions for Yellowbrick PyTest
"""

##########################################################################
## Imports
##########################################################################

import os


##########################################################################
## PyTest Hooks
##########################################################################

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
    if not hasattr(item.parent, 'obj'):
        return

    # Collect test objects to inspect
    parent = item.parent.obj
    node = item.obj

    # Goal: produce a parsable string of the relative path, parent docstring
    # or class name, and the docstring of the test case, then set the nodeid
    # so that pytest-spec will correctly parse the information.
    path = os.path.relpath(str(item.fspath))
    prefix = parent.__doc__ or getattr(parent, '__name__', parent.__class__.__name__)
    suffix = node.__doc__ if node.__doc__ else node.__name__

    if prefix or suffix:
        item._nodeid = '::'.join((path, prefix.strip(), suffix.strip()))
