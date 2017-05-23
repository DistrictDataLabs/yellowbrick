# tests.base
# Helper functions and cases for making assertions on visualizations.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Sun Oct 09 12:23:13 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: base.py [b8e3318] benjamin@bengfort.com $

"""
Helper functions and cases for making assertions on visualizations.
"""

##########################################################################
## Imports
##########################################################################

import inspect
import os

import unittest
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker

from matplotlib.testing.compare import compare_images
from matplotlib.testing.exceptions import ImageComparisonFailure


##########################################################################
## Visual Test Case
##########################################################################

class VisualTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(klass):
        """
        In order for tests to pass on Travis-CI we must use the 'Agg'
        matplotlib backend. This setup function ensures that all tests
        that do visual work setup the backend correctly.

        Note:
        """
        klass._backend = mpl.get_backend()
        super(VisualTestCase, klass).setUpClass()

    def setUp(self):
        """
        Assert tthat the backend is 'Agg' and close all previous plots
        """
        plt.close("all")
        from matplotlib import rcParams
        rcParams['font.family'] = 'DejaVu Sans'

        self.assertEqual(self._backend, 'agg')
        super(VisualTestCase, self).setUp()

    def _setup_imagetest(self, inspect_obj=None):
        if inspect_obj is not None:
            full_path = inspect_obj[1][1][:-3]
            self._module_path =  full_path.split('yellowbrick')[1].split('/')[2:]
            self._test_func_name = inspect_obj[1][3]
        return self._module_path, self._test_func_name

    def img_outpath(self, extension='.png'):
        module_path, test_func_name = self._setup_imagetest()
        module_path = os.path.join(*module_path)
        actual_images = os.path.join('tests', 'actual_images', module_path)

        if not os.path.exists(actual_images):
            mpl.cbook.mkdirs(actual_images)

        self._test_img_outpath = os.path.join(actual_images, test_func_name + extension)
        return self._test_img_outpath

    def _get_base_img(self, extension='.png'):
        module_path, test_func_name = self._setup_imagetest()

        module_path = os.path.join(*module_path)
        base_results = os.path.join('tests', 'baseline_images', module_path)

        if not os.path.exists(base_results):
            mpl.cbook.mkdirs(base_results)

        base_img = os.path.join(base_results, test_func_name + extension)

        return base_img

    def assert_images_similar(self, visualizer, tolerance=0.01):
        inspect_obj = inspect.stack()
        module_path, test_func_name = self._setup_imagetest(inspect_obj=inspect_obj)
        remove_ticks_and_titles(visualizer.ax)
        plt.savefig(self.img_outpath())
        base_image = self._get_base_img()
        test_img = self.img_outpath()
        yb_compare_images(base_image, test_img, tolerance)


def remove_ticks_and_titles(ax):
    # figure.suptitle("")
    null_formatter = ticker.NullFormatter()
    # for ax in figure.get_axes():
    ax.set_title("")
    ax.xaxis.set_major_formatter(null_formatter)
    ax.xaxis.set_minor_formatter(null_formatter)
    ax.yaxis.set_major_formatter(null_formatter)
    ax.yaxis.set_minor_formatter(null_formatter)
    try:
        ax.zaxis.set_major_formatter(null_formatter)
        ax.zaxis.set_minor_formatter(null_formatter)
    except AttributeError:
        pass



def yb_compare_images(expected, actual, tol):
    __tracebackhide__ = True

    if not os.path.exists(expected):
        raise ImageComparisonFailure('image does not exist: %s' % expected)

    err = compare_images(expected, actual, tol, in_decorator=True)

    if err:
        for key in ["actual", "expected"]:
            err[key] = os.path.relpath(err[key])
        raise ImageComparisonFailure(
            'images not close (RMS %(rms).3f):\n\t%(actual)s\n\t%(expected)s '
             % err)
