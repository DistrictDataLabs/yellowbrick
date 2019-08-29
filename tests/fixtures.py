# tests.fixtures
# Helpers for pytest fixtures and data related testing.
#
# Author:   Zijie (ZJ) Poh <zjpoh@noreply.github.com>
# Created:  Wed Feb 13 13:24:24 2019 -0400
#
# Copyright (C) 2019 The scikit-yb developers.
# For license information, see LICENSE.txt
#
# ID: fixtures.py [eb9f8cc] 8103276+zjpoh@users.noreply.github.com $

"""
Helpers for pytest fixtures and data related testing.
"""

##########################################################################
## Imports and Module Variables
##########################################################################

from collections import namedtuple


## Used for wrapping an dataset into a single variable.
Dataset = namedtuple("Dataset", "X,y")
Split = namedtuple("Split", "train,test")
