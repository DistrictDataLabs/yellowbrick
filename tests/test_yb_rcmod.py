# tests.test_utils
# Test the export module - to generate a corpus for machine learning.
#
# Author:    Patrick O'Melveny <pvomelveny@gmail.com>
# Created:  Thurs Jun 3
#
# For license information, see LICENSE.txt
#


##########################################################################
## Imports
##########################################################################
import unittest
import numpy.testing as npt
import numpy as np
import matplotlib as mpl
from distutils.version import LooseVersion


from yellowbrick import yb_rcmod


class RCParamTester(unittest.TestCase):

    excluded_params = {
        "backend",  # This cannot be changed by manipulating rc
        "svg.embed_char_paths"  # This param causes test issues and is deprecated anyway
    }

    def flatten_list(self, orig_list):

        iter_list = map(np.atleast_1d, orig_list)
        flat_list = [item for sublist in iter_list for item in sublist]
        return flat_list

    def assert_rc_params(self, params):

        for k, v in params.items():
            if k in self.excluded_params:
                continue
            elif isinstance(v, np.ndarray):
                npt.assert_array_equal(mpl.rcParams[k], v)
            else:
                self.assertEqual((k, mpl.rcParams[k]), (k, v))


class TestAxesStyle(RCParamTester):

    styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]

    def test_default_return(self):

        current = yb_rcmod.axes_style()
        self.assert_rc_params(current)

    def test_key_usage(self):

        _style_keys = set(yb_rcmod._style_keys)
        for style in self.styles:
            self.assertTrue(not set(yb_rcmod.axes_style(style)) ^ _style_keys)

    def test_bad_style(self):

        with self.assertRaises(ValueError):
            yb_rcmod.axes_style("i_am_not_a_style")

    def test_rc_override(self):

        rc = {"axes.facecolor": "blue", "foo.notaparam": "bar"}
        out = yb_rcmod.axes_style("darkgrid", rc)
        self.assertEqual(out["axes.facecolor"], "blue")
        self.assertNotIn("foo.notaparam", out)

    def test_set_style(self):

        for style in self.styles:

            style_dict = yb_rcmod.axes_style(style)
            yb_rcmod.set_style(style)
            self.assert_rc_params(style_dict)

    def test_style_context_manager(self):

        yb_rcmod.set_style("darkgrid")
        orig_params = yb_rcmod.axes_style()
        context_params = yb_rcmod.axes_style("whitegrid")

        with yb_rcmod.axes_style("whitegrid"):
            self.assert_rc_params(context_params)
        self.assert_rc_params(orig_params)

        @yb_rcmod.axes_style("whitegrid")
        def func():
            self.assert_rc_params(context_params)
        func()
        self.assert_rc_params(orig_params)

    def test_style_context_independence(self):

        self.assertTrue(set(yb_rcmod._style_keys) ^ set(yb_rcmod._context_keys))

    def test_set_rc(self):

        yb_rcmod.set(rc={"lines.linewidth": 4})
        self.assertEqual(mpl.rcParams["lines.linewidth"], 4)
        yb_rcmod.set()

    def test_reset_defaults(self):

        # Changes to the rc parameters make this test hard to manage
        # on older versions of matplotlib, so we'll skip it
        if LooseVersion(mpl.__version__) < LooseVersion("1.3"):
            raise self.SkipTest

        yb_rcmod.reset_defaults()
        self.assert_rc_params(mpl.rcParamsDefault)
        yb_rcmod.set()

    def test_reset_orig(self):

        # Changes to the rc parameters make this test hard to manage
        # on older versions of matplotlib, so we'll skip it
        if LooseVersion(mpl.__version__) < LooseVersion("1.3"):
            raise self.SkipTest

        yb_rcmod.reset_orig()
        self.assert_rc_params(mpl.rcParamsOrig)
        yb_rcmod.set()


class TestPlottingContext(RCParamTester):

    contexts = ["paper", "notebook", "talk", "poster"]

    def test_default_return(self):

        current = yb_rcmod.plotting_context()
        self.assert_rc_params(current)

    def test_key_usage(self):

        _context_keys = set(yb_rcmod._context_keys)
        for context in self.contexts:
            missing = set(yb_rcmod.plotting_context(context)) ^ _context_keys
            self.assertTrue(not missing)

    def test_bad_context(self):

        with self.assertRaises(ValueError):
            yb_rcmod.plotting_context("i_am_not_a_context")

    def test_font_scale(self):

        notebook_ref = yb_rcmod.plotting_context("notebook")
        notebook_big = yb_rcmod.plotting_context("notebook", 2)

        font_keys = ["axes.labelsize", "axes.titlesize", "legend.fontsize",
                     "xtick.labelsize", "ytick.labelsize", "font.size"]

        for k in font_keys:
            self.assertEqual(notebook_ref[k] * 2, notebook_big[k])

    def test_rc_override(self):

        key, val = "grid.linewidth", 5
        rc = {key: val, "foo": "bar"}
        out = yb_rcmod.plotting_context("talk", rc=rc)
        self.assertEqual(out[key], val)
        self.assertNotIn("foo", out)

    def test_set_context(self):

        for context in self.contexts:

            context_dict = yb_rcmod.plotting_context(context)
            yb_rcmod.set_context(context)
            self.assert_rc_params(context_dict)

    def test_context_context_manager(self):

        yb_rcmod.set_context("notebook")
        orig_params = yb_rcmod.plotting_context()
        context_params = yb_rcmod.plotting_context("paper")

        with yb_rcmod.plotting_context("paper"):
            self.assert_rc_params(context_params)
        self.assert_rc_params(orig_params)

        @yb_rcmod.plotting_context("paper")
        def func():
            self.assert_rc_params(context_params)
        func()
        self.assert_rc_params(orig_params)


if __name__ == "__main__":
    unittest.main()
