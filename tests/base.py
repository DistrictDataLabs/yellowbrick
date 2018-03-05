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
from matplotlib import rcParams

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
        # Reset the matplotlib environment
        plt.cla()        # clear current axis
        plt.clf()        # clear current figure
        plt.close("all") # close all existing plots

        # Travis-CI does not have san-serif
        rcParams['font.family'] = 'DejaVu Sans'

        # Assert that the backend is agg
        self.assertEqual(self._backend, 'agg')
        super(VisualTestCase, self).setUp()

    def _setup_imagetest(self, inspect_obj=None):
        """Parses the module path and test function name from an inspect call obj
        that is triggered in the unittest specific "assert_images_similar"
        """
        if inspect_obj is not None:
            full_path = inspect_obj[1][1][:-3]
            self._module_path =  full_path.split('yellowbrick')[1].split('/')[2:]
            self._test_func_name = inspect_obj[1][3]
        return self._module_path, self._test_func_name

    def _actual_img_path(self, extension='.png'):
        """Determines the correct outpath for drawing a matplotlib image that
        corresponds to the unittest module path.
        """
        module_path, test_func_name = self._setup_imagetest()
        module_path = os.path.join(*module_path)
        actual_images = os.path.join('tests', 'actual_images', module_path)

        if not os.path.exists(actual_images):
            mpl.cbook.mkdirs(actual_images)

        self._test_img_outpath = os.path.join(actual_images, test_func_name + extension)
        return self._test_img_outpath

    def _base_img_path(self, extension='.png'):
        """Gets the baseline_image path for comparison that corresponds to the
        unittest module path.
        """

        module_path, test_func_name = self._setup_imagetest()
        module_path = os.path.join(*module_path)
        base_results = os.path.join('tests', 'baseline_images', module_path)
        if not os.path.exists(base_results):
            mpl.cbook.mkdirs(base_results)
        base_img = os.path.join(base_results, test_func_name + extension)
        return base_img

    def assert_images_similar(self, visualizer=None, ax=None, tol=0.01):
        """Accessible testing method for testing generation of a Visualizer.

        Requires the placement of a baseline image for comparison in the
        tests/baseline_images folder that corresponds to the module path of the
        VisualTestCase being evaluated. The name of the image corresponds to
        the unittest function where "self.assert_images_similar" is called.

        For example, calling "assert_images_similar" in the unittest
        "test_class_report" in tests.test_classifier.test_class_balance would
        require placement a baseline image at:

        baseline_images/test_classifier/test_class_balance/test_class_report.png

        The easiest way to generate a baseline image is to first run the test that
        calls "assert_images_similar", and then copy the actual test generated
        image from:

        actual_images/

        visualizer : yellowbrick visualizer
            An instantiated yellowbrick visualizer that has been fitted,
            transformed and had all operations except for poof called on it.

        ax : matplotlib Axes, default: None
            The axis to plot the figure on.

        tol : float
            The tolerance (a color value difference, where 255 is the
            maximal difference).  The test fails if the average pixel
            difference is greater than this value.
        """
        if visualizer is None and ax is None:
            raise ValueError("must supply either a visualizer or axes")

        ax = ax or visualizer.ax

        # inspect is used to locate and organize the baseline images and actual
        # test generated images for comparison
        inspect_obj = inspect.stack()
        module_path, test_func_name = self._setup_imagetest(inspect_obj=inspect_obj)

        # clean and remove the textual/ formatting elements from the visualizer
        remove_ticks_and_titles(ax)

        plt.savefig(self._actual_img_path())
        base_image = self._base_img_path()
        test_img = self._actual_img_path()
        # test it!
        yb_compare_images(base_image, test_img, tol)


def remove_ticks_and_titles(ax):
    """Removes tickets and formatting on sub ax object that is useful for the
    assert_images_similar as different OS having varying font styles and other
    system level differences
    """
    null_formatter = ticker.NullFormatter()
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
    """ Compares a baseline image and test generated actual image
    using a matplotlib's built-in imagine comparison function

    expected : string, imagepath
        The image filepath to the baseline image

    actual : string, imagepath
        The image filepath to the actual test generated image

    tol : float
        The tolerance (a color value difference, where 255 is the
        maximal difference).  The test fails if the average pixel
        difference is greater than this value.
    """
    __tracebackhide__ = True

    if not os.path.exists(expected):
        raise ImageComparisonFailure('image does not exist: %s' % expected)

    # method from matplotlib.testing.compare
    err = compare_images(expected, actual, tol, in_decorator=True)

    if err:
        raise ImageComparisonFailure(
            'images not close (RMS %(rms).3f):\n\t%(actual)s\n\t%(expected)s '
             % err)
