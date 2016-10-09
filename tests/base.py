# tests.base
# Helper functions and cases for making assertions on visualizations.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Sun Oct 09 12:23:13 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: base.py [] benjamin@bengfort.com $

"""
Helper functions and cases for making assertions on visualizations.
"""

##########################################################################
## Imports
##########################################################################

import unittest
import matplotlib as mpl


##########################################################################
## Visual Test Case
##########################################################################

class VisualTestCase(unittest.TestCase):

    def beforeTestSuite(self):
        mpl.use('Agg')
