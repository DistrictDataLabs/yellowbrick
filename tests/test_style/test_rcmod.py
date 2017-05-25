# tests.test_style.test_rcmod
# Testing the matplotlib configuration modifications for aesthetic.
#
# Author:   Patrick O'Melveny <pvomelveny@gmail.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Oct 06 08:20:33 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_rcmod.py [c6aff34] benjamin@bengfort.com $

"""
Testing the matplotlib configuration modifications for aesthetic.
"""

##########################################################################
## Imports
##########################################################################

import unittest
import numpy as np
import matplotlib as mpl
import numpy.testing as npt
import yellowbrick.style.rcmod as yb_rcmod

from distutils.version import LooseVersion
from tests.base import VisualTestCase


##########################################################################
## Parameter Tests
##########################################################################

class RCParamTester(VisualTestCase):
    """
    Base class for asserting parameters have been correctly changed.
    """

    excluded_params = {
        "backend",  # This cannot be changed by manipulating rc
        "svg.embed_char_paths",  # This param causes test issues and is deprecated anyway
        "font.family", # breaks the visualtest case
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


##########################################################################
## Parameter Tests
##########################################################################

class TestAxesStyle(RCParamTester):

    def test_default_return(self):
        """
        Test that the axes style returns the default params
        """
        current = yb_rcmod._axes_style()
        self.assert_rc_params(current)

    def test_rc_override(self):
        """
        Test being able to override the rc params
        """

        rc = {"axes.facecolor": "blue", "foo.notaparam": "bar"}
        out = yb_rcmod._axes_style("darkgrid", rc)
        self.assertEqual(out["axes.facecolor"], "blue")
        self.assertNotIn("foo.notaparam", out)

    def test_set_style(self):
        """
        Test setting the yellowbrick axes style
        """
        style_dict = yb_rcmod._axes_style()
        yb_rcmod.set_style()
        self.assert_rc_params(style_dict)

    @unittest.skip("This test doesn't make sense without multiple styles")
    def test_style_context_manager(self):

        yb_rcmod.set_style("darkgrid")
        orig_params = yb_rcmod._axes_style()
        context_params = yb_rcmod._axes_style("whitegrid")

        with yb_rcmod._axes_style("whitegrid"):
            self.assert_rc_params(context_params)
        self.assert_rc_params(orig_params)

        @yb_rcmod._axes_style("whitegrid")
        def func():
            self.assert_rc_params(context_params)
        func()
        self.assert_rc_params(orig_params)

    def test_style_context_independence(self):
        """
        Assert context and style independence
        """
        self.assertTrue(set(yb_rcmod._style_keys) ^ set(yb_rcmod._context_keys))

    def test_set_rc(self):
        """
        Test the ability to set the mpl configuration rc dict
        """
        yb_rcmod.set_aesthetic(rc={"lines.linewidth": 4})
        self.assertEqual(mpl.rcParams["lines.linewidth"], 4)
        yb_rcmod.set_aesthetic()

    def test_reset_defaults(self):
        """
        Test the ability to reset to the mpl defaults
        """
        # Changes to the rc parameters make this test hard to manage
        # on older versions of matplotlib, so we'll skip it
        if LooseVersion(mpl.__version__) < LooseVersion("1.3"):
            raise self.SkipTest

        yb_rcmod.reset_defaults()
        self.assert_rc_params(mpl.rcParamsDefault)
        yb_rcmod.set_aesthetic()

    def test_reset_orig(self):
        """
        Test the ability to reset to the original (respecting custom styles)
        """

        # Changes to the rc parameters make this test hard to manage
        # on older versions of matplotlib, so we'll skip it
        if LooseVersion(mpl.__version__) < LooseVersion("1.3"):
            raise self.SkipTest

        yb_rcmod.reset_orig()
        self.assert_rc_params(mpl.rcParamsOrig)
        yb_rcmod.set_aesthetic()


class TestPlottingContext(RCParamTester):

    def test_default_return(self):
        """
        Test the context returns the default
        """
        current = yb_rcmod._plotting_context()
        self.assert_rc_params(current)

    def test_font_scale(self):
        """
        Test scaling the fonts
        """

        notebook_ref = yb_rcmod._plotting_context("notebook")
        notebook_big = yb_rcmod._plotting_context("notebook", 2)

        font_keys = ["axes.labelsize", "axes.titlesize", "legend.fontsize",
                     "xtick.labelsize", "ytick.labelsize", "font.size"]

        for k in font_keys:
            self.assertEqual(notebook_ref[k] * 2, notebook_big[k])

    def test_rc_override(self):
        """
        Test overriding the rc params dictionary
        """
        key, val = "grid.linewidth", 5
        rc = {key: val, "foo": "bar"}
        out = yb_rcmod._plotting_context("talk", rc=rc)
        self.assertEqual(out[key], val)
        self.assertNotIn("foo", out)

    def test__set_context(self):
        """
        Test setting the context
        """
        context_dict = yb_rcmod._plotting_context()
        yb_rcmod._set_context()
        self.assert_rc_params(context_dict)

    @unittest.skip("This test doesn't make sense without multiple contexts")
    def test_context_context_manager(self):

        yb_rcmod._set_context("notebook")
        orig_params = yb_rcmod._plotting_context()
        context_params = yb_rcmod._plotting_context("paper")

        with yb_rcmod._plotting_context("paper"):
            self.assert_rc_params(context_params)
        self.assert_rc_params(orig_params)

        @yb_rcmod._plotting_context("paper")
        def func():
            self.assert_rc_params(context_params)
        func()
        self.assert_rc_params(orig_params)


if __name__ == "__main__":
    unittest.main()
