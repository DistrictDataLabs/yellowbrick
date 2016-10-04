# tests.test_colors.test_palettes
# Tests the palettes module of the yellowbrick library.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Tue Oct 04 16:21:58 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_palettes.py [] benjamin@bengfort.com $

"""
Tests the palettes module of the yellowbrick library.
"""

##########################################################################
## Imports
##########################################################################

import unittest
import matplotlib as mpl

from yellowbrick.exceptions import *
from yellowbrick.colors.palettes import *
from yellowbrick.colors.palettes import ColorPalette, PALETTES


##########################################################################
## Color Palette Tests
##########################################################################

class ColorPaletteTests(unittest.TestCase):

    def test_init_palette_by_name(self):
        """
        Test that a palette can be initialized by name
        """

        # Try all the names in the palettes
        for name, value in PALETTES.items():
            try:
                palette = ColorPalette(name)
            except YellowbrickValueError:
                self.fail(
                    "Could not instantiate {} color palette by name".format(name)
                )

            self.assertEqual(value, palette)

        # Try a name not in PALETTES
        with self.assertRaises(YellowbrickValueError):
            self.assertNotIn('foo', PALETTES, "Cannot test bad name 'foo' it is in PALETTES!")
            palette = ColorPalette('foo')

    def test_init_palette_by_list(self):
        """
        Test that a palette can be initialized by a list
        """

        # Try all the values in the palettes (HEX)
        for value in PALETTES.values():
            palette = ColorPalette(value)
            self.assertEqual(len(value), len(palette))

        # Try all the values converted to RGB
        for value in PALETTES.values():
            palette = ColorPalette(map(mpl.colors.colorConverter.to_rgb, value))
            self.assertEqual(len(value), len(palette))
