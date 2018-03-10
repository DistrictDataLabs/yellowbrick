# tests.test_style.test_colors
# Tests for the color utilities and helpers module
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Oct 06 09:30:49 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_colors.py [c6aff34] benjamin@bengfort.com $

"""
Tests for the color utilities and helper functions
"""

##########################################################################
## Imports
##########################################################################

import pytest

from matplotlib import cm
from cycler import Cycler

from yellowbrick.style.colors import *
from yellowbrick.style.palettes import ColorPalette, PALETTES

from tests.base import VisualTestCase


##########################################################################
## Color Tests
##########################################################################

class TestGetColorCycle(VisualTestCase):
    """
    Test get_color_cycle helper function
    """

    def test_cycle_depends_on_palette(self):
        """
        Ensure the color cycle depends on the palette
        """
        c = get_color_cycle()
        assert len(c) == 6

        with ColorPalette('paired'):
            c = get_color_cycle()
            assert len(c) == 12

        c = get_color_cycle()
        assert len(c) == 6

    @pytest.mark.filterwarnings()
    @pytest.mark.skipif(not mpl_ge_150, reason="requires matplotlib 1.5 or later")
    def test_mpl_ge_150(self):
        """
        Test get color cycle with matplotlib 1.5 or later
        """
        colors = get_color_cycle()
        cycle = mpl.rcParams['axes.prop_cycle']

        # Ensure the cycle is in fact a cycle
        assert isinstance(cycle, Cycler)

        # Ensure that colors is actually a list (might change in the future)
        assert isinstance(colors, list)

        # Ensure the cycler and the colors have the same length
        cycle = list(cycle)
        assert len(colors) == len(cycle)

        # Ensure the colors and cycle match
        for color, cycle_color in zip(colors, cycle):
            assert color == cycle_color['color']


    @pytest.mark.filterwarnings()
    @pytest.mark.skipif(mpl_ge_150, reason="requires matplotlib ealier than 1.5")
    def test_mpl_lt_150(self):
        """
        Test get color cycle with matplotlib earlier than 1.5
        """
        assert get_color_cycle() == mpl.rcParams['axes.color_cycle']


class TestResolveColors(VisualTestCase):
    """
    Test resolve_colors helper function
    """

    def test_resolve_colors_default(self):
        """
        Provides reasonable defaults provided no arguments
        """
        colors = resolve_colors()
        assert colors == get_color_cycle()

    def test_resolve_colors_default_truncate(self):
        """
        Truncates default colors when n_colors is smaller than palette
        """
        assert len(get_color_cycle()) > 3
        assert len(resolve_colors(3)) == 3

    def test_resolve_colors_default_multiply(self):
        """
        Multiplies default colors when n_colors is larger than palette
        """
        assert len(get_color_cycle()) < 18
        assert len(resolve_colors(18)) == 18

    def test_warning_on_colormap_and_colors_args(self):
        """
        Warns when both colormap and colors is used, colors is default
        """
        with pytest.warns(Warning, match="both colormap and colors specified"):
            colors = resolve_colors(colormap='RdBu', colors=['r', 'g', 'b'])
            assert colors == ['r', 'g', 'b']

    def test_colormap_invalid(self):
        """
        Exception raised when invalid colormap is supplied
        """
        with pytest.raises(YellowbrickValueError):
            resolve_colors(12, colormap='foo')

    def test_colormap_string(self):
        """
        Check resolve colors works when a colormap string is passed
        """
        cases = (
            (
                {'n_colors': 6, 'colormap': 'RdBu'},
                [
                    (0.403921568627451, 0.0, 0.12156862745098039, 1.0),
                    (0.8392156862745098, 0.3764705882352941, 0.30196078431372547, 1.0),
                    (0.9921568627450981, 0.8588235294117647, 0.7803921568627451, 1.0),
                    (0.8196078431372551, 0.8980392156862746, 0.9411764705882353, 1.0),
                    (0.2627450980392157, 0.5764705882352941, 0.7647058823529411, 1.0),
                    (0.0196078431372549, 0.18823529411764706, 0.3803921568627451, 1.0)
                ],
            ),
            (
                {'n_colors': 18, 'colormap': 'viridis'},
                [
                    (0.267004, 0.004874, 0.329415, 1.0),
                    (0.281924, 0.089666, 0.412415, 1.0),
                    (0.280255, 0.165693, 0.476498, 1.0),
                    (0.263663, 0.237631, 0.518762, 1.0),
                    (0.237441, 0.305202, 0.541921, 1.0),
                    (0.208623, 0.367752, 0.552675, 1.0),
                    (0.182256, 0.426184, 0.55712, 1.0),
                    (0.159194, 0.482237, 0.558073, 1.0),
                    (0.13777, 0.537492, 0.554906, 1.0),
                    (0.121148, 0.592739, 0.544641, 1.0),
                    (0.128087, 0.647749, 0.523491, 1.0),
                    (0.180653, 0.701402, 0.488189, 1.0),
                    (0.274149, 0.751988, 0.436601, 1.0),
                    (0.395174, 0.797475, 0.367757, 1.0),
                    (0.535621, 0.835785, 0.281908, 1.0),
                    (0.688944, 0.865448, 0.182725, 1.0),
                    (0.845561, 0.887322, 0.099702, 1.0),
                    (0.993248, 0.906157, 0.143936, 1.0)
                ],
            ),
            (
                {'n_colors': 9, 'colormap': 'Set1'},
                [
                    (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0),
                    (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0),
                    (0.30196078431372547, 0.6862745098039216, 0.2901960784313726, 1.0),
                    (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0),
                    (1.0, 0.4980392156862745, 0.0, 1.0),
                    (1.0, 1.0, 0.2, 1.0),
                    (0.6509803921568628, 0.33725490196078434, 0.1568627450980392, 1.0),
                    (0.9686274509803922, 0.5058823529411764, 0.7490196078431373, 1.0),
                    (0.6, 0.6, 0.6, 1.0)
                ],
            ),
        )

        for kwds, expected in cases:
            colors = resolve_colors(**kwds)
            assert isinstance(colors, list)
            assert colors == expected

    def test_colormap_string_default_length(self):
        """
        Check colormap when n_colors is not specified
        """
        n_colors = len(get_color_cycle())
        assert len(resolve_colors(colormap='autumn')) == n_colors

    def test_colormap_cmap(self):
        """
        Assert that supplying a maptlotlib.cm as colormap works
        """
        cmap = cm.get_cmap('nipy_spectral')
        colors = resolve_colors(4, colormap=cmap)
        assert colors == [
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.6444666666666666, 0.7333666666666667, 1.0),
            (0.7999666666666666, 0.9777666666666667, 0.0, 1.0),
            (0.8, 0.8, 0.8, 1.0)
        ]

    def test_colors(self):
        """
        Test passing in a list of colors
        """
        c = PALETTES['flatui']
        assert resolve_colors(colors=c) == c

    def test_colors_truncate(self):
        """
        Test passing in a list of colors with n_colors truncate
        """
        c = PALETTES['flatui']

        assert len(c) > 3
        assert len(resolve_colors(n_colors=3, colors=c)) == 3

    def test_colors_multiply(self):
        """
        Test passing in a list of colors with n_colors multiply
        """
        c = PALETTES['flatui']

        assert len(c) < 12
        assert len(resolve_colors(n_colors=12, colors=c)) == 12
