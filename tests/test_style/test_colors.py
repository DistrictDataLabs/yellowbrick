# tests.test_style.test_colors
# Tests for the color utilities and helpers module
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Oct 06 09:30:49 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_colors.py [] benjamin@bengfort.com $

"""
Tests for the color utilities and helpers module
"""

##########################################################################
## Imports
##########################################################################

import warnings
import unittest

from yellowbrick.style import *
from yellowbrick.style.colors import *
from tests.base import VisualTestCase


##########################################################################
## Color Tests
##########################################################################

class ColorUtilitiesTests(VisualTestCase):

    def test_get_color_cycle(self):
        """
        Test the retreival of the current color cycle
        """
        c = get_color_cycle()
        self.assertEqual(len(c), 6)

        set_palette('paired')
        c = get_color_cycle()
        self.assertEqual(len(c), 12)
