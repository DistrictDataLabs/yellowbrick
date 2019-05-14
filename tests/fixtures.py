# tests.fixtures
# Helpers for pytest fixtures and data related testing.
#
# Author:   Zijie (ZJ) Poh <zjpoh@noreply.github.com>
# Created:  Wed Feb 13 13:24:24 2019 -0400
#
# Copyright (C) 2017 The scikit-yb developers.
# For license information, see LICENSE.txt
#
# ID: fixtures.py [] benjamin@bengfort.com $

"""
Helpers for pytest fixtures and data related testing.
"""

##########################################################################
## Imports and Module Variables
##########################################################################

from collections import namedtuple


## Used for wrapping an dataset into a single variable.
TestDataset = namedtuple('TestDataset', 'X,y')
