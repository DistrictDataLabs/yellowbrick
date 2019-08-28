# tests.test_style.test_palettes
# Tests the palettes module of the yellowbrick library.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Tue Oct 04 16:21:58 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_palettes.py [c6aff34] benjamin@bengfort.com $

"""
Tests the palettes module of the yellowbrick library.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np
import matplotlib as mpl

from yellowbrick.exceptions import *
from yellowbrick.style.palettes import *
from yellowbrick.style.colors import get_color_cycle
from yellowbrick.style.rcmod import set_aesthetic, set_palette
from yellowbrick.style.palettes import color_sequence, color_palette
from yellowbrick.style.palettes import ColorPalette, PALETTES, SEQUENCES

from tests.base import VisualTestCase


##########################################################################
## Color Palette Tests
##########################################################################


class TestColorPaletteObject(VisualTestCase):
    """
    Tests the ColorPalette object
    """

    def test_init_palette_by_name(self):
        """
        Test that a palette can be initialized by name
        """

        # Try all the names in the palettes
        for name, value in PALETTES.items():
            try:
                palette = ColorPalette(name)
            except YellowbrickValueError:
                self.fail("Could not instantiate {} color palette by name".format(name))

            assert value == palette

        # Try a name not in PALETTES
        with pytest.raises(YellowbrickValueError):
            assert (
                "foo" not in PALETTES
            ), "Cannot test bad name 'foo' it is in PALETTES!"
            palette = ColorPalette("foo")

    def test_init_palette_by_list(self):
        """
        Test that a palette can be initialized by a list
        """

        # Try all the values in the palettes (HEX)
        for value in PALETTES.values():
            palette = ColorPalette(value)
            assert len(value) == len(palette)

        # Try all the values converted to RGB
        for value in PALETTES.values():
            palette = ColorPalette(map(mpl.colors.colorConverter.to_rgb, value))
            assert len(value) == len(palette)

    def test_color_palette_context(self):
        """
        Test ColorPalette context management
        """
        default = color_palette()
        context = color_palette("dark")

        with ColorPalette("dark") as palette:
            assert isinstance(palette, ColorPalette)
            assert get_color_cycle() == context

        assert get_color_cycle() == default

    def test_as_hex_as_rgb(self):
        """
        Test the conversion of a ColorPalette to hex values and back to rgb
        """
        palette = color_palette("flatui")
        expected = PALETTES["flatui"]
        morgified = palette.as_hex()

        assert morgified is not palette
        assert isinstance(morgified, ColorPalette)
        assert morgified == expected

        remorgified = morgified.as_rgb()
        assert remorgified is not morgified
        assert remorgified is not palette
        assert remorgified == palette

    @pytest.mark.skip(reason="not implemented yet")
    def test_plot_color_palette(self):
        """
        Test the plotting of a color palette for color visualization
        """
        raise NotImplementedError("Not quite sure how to implement this yet")


class TestColorPaletteFunction(VisualTestCase):
    """
    Tests the color_palette function.
    """

    def test_current_palette(self):
        """
        Test modifying the current palette with a simple palette
        """
        pal = color_palette(["red", "blue", "green"], 3)
        set_palette(pal, 3)
        assert pal == get_color_cycle()

        # Reset the palette
        set_aesthetic()

    def test_palette_context(self):
        """
        Test the context manager for the color_palette function
        """

        default_pal = color_palette()
        context_pal = color_palette("muted")

        with color_palette(context_pal):
            assert get_color_cycle() == context_pal

        assert get_color_cycle() == default_pal

    def test_big_palette_context(self):
        """
        Test that the context manager also resets the number of colors
        """

        original_pal = color_palette("accent", n_colors=8)
        context_pal = color_palette("bold", 10)

        set_palette(original_pal)
        with color_palette(context_pal, 10):
            assert get_color_cycle() == context_pal

        assert get_color_cycle() == original_pal

        # Reset default
        set_aesthetic()

    def test_yellowbrick_palettes(self):
        """
        Test the yellowbrick palettes have length 6 (bgrmyck)
        """
        pals = ["accent", "dark", "pastel", "bold", "muted"]
        for name in pals:
            pal_out = color_palette(name)
            assert len(pal_out) == 6, "{} is not of len 6".format(name)

    def test_seaborn_palettes(self):
        """
        Test the seaborn palettes have length 6 (bgrmyck)
        """
        pals = [
            "sns_deep",
            "sns_muted",
            "sns_pastel",
            "sns_bright",
            "sns_dark",
            "sns_colorblind",
        ]
        for name in pals:
            pal_out = color_palette(name)
            assert len(pal_out) == 6

    def test_other_palettes(self):
        """
        Test that the other palettes exist
        """
        pals = ["flatui", "paired", "neural_paint", "set1"]
        for name in pals:
            pal_out = color_palette(name)
            assert pal_out is not None
            assert len(pal_out) > 0

    def test_bad_palette_name(self):
        """
        Test that a bad palette name raises an exception
        """

        with pytest.raises(ValueError):
            color_palette("IAmNotAPalette")

        with pytest.raises(YellowbrickValueError):
            color_palette("IAmNotAPalette")

    def test_bad_palette_colors(self):
        """
        Test that bad color names raise an exception
        """

        pal = ["red", "blue", "iamnotacolor"]
        with pytest.raises(ValueError):
            color_palette(pal)

        with pytest.raises(YellowbrickValueError):
            color_palette(pal)

    def test_palette_is_list_of_tuples(self):
        """
        Assert that color_palette returns a list of RGB tuples
        """

        pal_in = np.array(["red", "blue", "green"])
        pal_out = color_palette(pal_in, 3)

        assert isinstance(pal_out, list)
        assert isinstance(pal_out[0], tuple)
        assert isinstance(pal_out[0][0], float)
        assert len(pal_out[0]) == 3

    def test_palette_cycles(self):
        """
        Test that the color palette cycles for more colors
        """
        accent = color_palette("accent")
        double_accent = color_palette("accent", 12)
        assert double_accent == accent + accent

    @pytest.mark.skip(reason="discovered this commented out, don't know why")
    def test_cbrewer_qual(self):
        """
        Test colorbrewer qualitative palettes
        """
        pal_short = mpl_palette("Set1", 4)
        pal_long = mpl_palette("Set1", 6)
        assert pal_short == pal_long[:4]

        pal_full = palettes.mpl_palette("Set2", 8)
        pal_long = palettes.mpl_palette("Set2", 10)
        assert pal_full == pal_long[:8]

    def test_color_codes(self):
        """
        Test the setting of color codes
        """
        set_color_codes("accent")
        colors = color_palette("accent") + ["0.06666666666666667"]
        for code, color in zip("bgrmyck", colors):
            rgb_want = mpl.colors.colorConverter.to_rgb(color)
            rgb_got = mpl.colors.colorConverter.to_rgb(code)
            assert rgb_want == rgb_got
        set_color_codes("reset")

    def test_as_hex(self):
        """
        Test converting a color palette to hex and back to rgb.
        """
        pal = color_palette("accent")
        for rgb, hex in zip(pal, pal.as_hex()):
            assert mpl.colors.rgb2hex(rgb) == hex

        for rgb_e, rgb_v in zip(pal, pal.as_hex().as_rgb()):
            assert rgb_e == rgb_v

    def test_preserved_palette_length(self):
        """
        Test palette length is preserved when modified
        """
        pal_in = color_palette("Set1", 10)
        pal_out = color_palette(pal_in)
        assert pal_in == pal_out

    def test_color_sequence(self):
        """
        Ensure the color sequence returns listed colors.
        """
        for name, ncols in SEQUENCES.items():
            for n in ncols.keys():
                cmap = color_sequence(name, n)
                assert name == cmap.name
                assert n == cmap.N

    def test_color_sequence_default(self):
        """
        Assert the default color sequence is RdBu
        """
        cmap = color_sequence()
        assert cmap.name == "RdBu"
        assert cmap.N == 11

    def test_color_sequence_unrecocognized(self):
        """
        Test value errors for unrecognized sequences
        """
        with pytest.raises(YellowbrickValueError):
            color_sequence("PepperBucks", 3)

    def test_color_sequence_bounds(self):
        """
        Test color sequence out of bounds value error
        """
        with pytest.raises(YellowbrickValueError):
            color_sequence("RdBu", 18)

        with pytest.raises(YellowbrickValueError):
            color_sequence("RdBu", 2)
