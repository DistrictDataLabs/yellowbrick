# tests
# Testing package for the yellowbrick visualization library.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  Wed May 18 10:48:46 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: __init__.py [0c5ba04] benjamin@bengfort.com $

"""
Testing package for the yellowbrick visualization library.
"""

##########################################################################
## Imports
##########################################################################

import unittest
import matplotlib

## IMPORTANT! Set matplotlib to use the Agg backend before imported anywhere!
matplotlib.use('Agg')


##########################################################################
## Test Constants
##########################################################################

EXPECTED_VERSION = "0.9"


##########################################################################
## Initialization Tests
##########################################################################

class InitializationTests(unittest.TestCase):

    def test_sanity(self):
        """
        Test that tests work by confirming 7-3 = 4
        """
        self.assertEqual(7-3, 4, "The world went wrong!!")

    def test_import(self):
        """
        Assert that the yellowbrick package can be imported.
        """
        try:
            import yellowbrick
        except ImportError:
            self.fail("Could not import the yellowbrick library!")

    def test_version(self):
        """
        Assert that the test version matches the library version.
        """
        try:
            import yellowbrick as yb
            self.assertEqual(yb.__version__, EXPECTED_VERSION)
        except ImportError:
            self.fail("Could not import the yellowbrick library!")
