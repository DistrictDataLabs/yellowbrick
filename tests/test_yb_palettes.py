# tests.test_yb_palettes
# Test the export module - to generate a corpus for machine learning.
#
# Author:   Patrick O'Melveny <pvomelveny@gmail.com>
# Created:  Friaday Jun 3
#
# For license information, see LICENSE.txt
#


##########################################################################
## Imports
##########################################################################
import unittest
import numpy as np
import matplotlib as mpl
import warnings


from yellowbrick import yb_palettes, yb_rcmod, color_utils


class TestColorPalettes(unittest.TestCase):

    def test_current_palette(self):

        pal = yb_palettes.color_palette(["red", "blue", "green"], 3)
        yb_rcmod.set_palette(pal, 3)
        self.assertEqual(pal, color_utils.get_color_cycle())
        yb_rcmod.set()

    def test_palette_context(self):

        default_pal = yb_palettes.color_palette()
        context_pal = yb_palettes.color_palette("muted")

        with yb_palettes.color_palette(context_pal):
            self.assertEqual(color_utils.get_color_cycle(), context_pal)

        self.assertEqual(color_utils.get_color_cycle(), default_pal)

    def test_big_palette_context(self):

        original_pal = yb_palettes.color_palette("accent", n_colors=8)
        context_pal = yb_palettes.color_palette("bold", 10)

        yb_rcmod.set_palette(original_pal)
        with yb_palettes.color_palette(context_pal, 10):
            self.assertEqual(color_utils.get_color_cycle(), context_pal)

        self.assertEqual(color_utils.get_color_cycle(), original_pal)

        # Reset default
        yb_rcmod.set()

    def test_yellowbrick_palettes(self):

        pals = ["accent", "dark", "paired", "pastel", "bold", "muted"]
        for name in pals:
            pal_out = yb_palettes.color_palette(name)
            self.assertEqual(len(pal_out), 6 if name != 'paired' else 10)

    def test_seaborn_palettes(self):

        pals = ["sns_deep", "sns_muted", "sns_pastel",
                "sns_bright", "sns_dark", "sns_colorblind"]
        for name in pals:
            pal_out = yb_palettes.color_palette(name)
            self.assertEqual(len(pal_out), 6)

    def test_bad_palette_name(self):

        with self.assertRaises(ValueError):
            yb_palettes.color_palette("IAmNotAPalette")

    def test_terrible_palette_name(self):

        with self.assertRaises(ValueError):
            yb_palettes.color_palette("jet")

    def test_bad_palette_colors(self):

        pal = ["red", "blue", "iamnotacolor"]
        with self.assertRaises(ValueError):
            yb_palettes.color_palette(pal)

    def test_palette_is_list_of_tuples(self):

        pal_in = np.array(["red", "blue", "green"])
        pal_out = yb_palettes.color_palette(pal_in, 3)

        self.assertIsInstance(pal_out, list)
        self.assertIsInstance(pal_out[0], tuple)
        self.assertIsInstance(pal_out[0][0], float)
        self.assertEqual(len(pal_out[0]), 3)

    def test_palette_cycles(self):

        accent = yb_palettes.color_palette("accent")
        double_accent = yb_palettes.color_palette("accent", 12)
        self.assertEqual(double_accent, accent + accent)

    """
    def test_cbrewer_qual(self):

        pal_short = yb_palettes.mpl_palette("Set1", 4)
        pal_long = yb_palettes.mpl_palette("Set1", 6)
        self.assertEqual(pal_short, pal_long[:4])

        pal_full = palettes.mpl_palette("Set2", 8)
        pal_long = palettes.mpl_palette("Set2", 10)
        self.assertEqual(pal_full, pal_long[:8])
    """

    def test_color_codes(self):

        yb_palettes.set_color_codes("accent")
        colors = yb_palettes.color_palette("accent") + [".1"]
        for code, color in zip("bgrmyck", colors):
            rgb_want = mpl.colors.colorConverter.to_rgb(color)
            rgb_got = mpl.colors.colorConverter.to_rgb(code)
            self.assertEqual(rgb_want, rgb_got)
        yb_palettes.set_color_codes("reset")

    def test_as_hex(self):

        pal = yb_palettes.color_palette("accent")
        for rgb, hex in zip(pal, pal.as_hex()):
            self.assertEqual(mpl.colors.rgb2hex(rgb), hex)
    """
    def test_preserved_palette_length(self):

        pal_in = palettes.color_palette("Set1", 10)
        pal_out = palettes.color_palette(pal_in)
        nt.assert_equal(pal_in, pal_out)
    """

    def test_get_color_cycle(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = color_utils.get_color_cycle()
            expected = mpl.rcParams['axes.color_cycle']
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
