# tests.base
# Helper functions and cases for making assertions on visualizations.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Sun Oct 09 12:23:13 2016 -0400
#
# ID: base.py [b8e3318] benjamin@bengfort.com $

"""
Helper functions and cases for making assertions on visualizations.
"""

##########################################################################
## Imports
##########################################################################

import os
import sys
import inspect

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib.testing.compare import compare_images
from yellowbrick.exceptions import ImageComparisonFailure


##########################################################################
## Environment
##########################################################################


def is_windows_or_conda():
    """
    Simple detection mechanism to determine if the tests are running in a
    win32 or Anaconda/Miniconda environment.
    """
    is_windows = sys.platform == "win32"
    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    return is_windows or is_conda


##########################################################################
## Module Constants
##########################################################################

# The root directory for all tests
TESTS = os.path.dirname(__file__)
ACTUAL_IMAGES = os.path.join(TESTS, "actual_images")
BASELINE_IMAGES = os.path.join(TESTS, "baseline_images")
IS_WINDOWS_OR_CONDA = is_windows_or_conda()


##########################################################################
## Visual Test Case
##########################################################################


class VisualTestCase(object):
    """
    The visual test case class ensures that all tests inside of the class
    can execute image similarity tests inside of a clean matplotlib global
    figure.
    """

    def setup_method(self):
        """
        Before a visual test case method is run, ensure that the previous
        figure is closed and the current axes are cleared.

        See: https://docs.pytest.org/en/latest/xunit_setup.html
        """
        # Reset the matplotlib environment
        plt.cla()  # clear current axis
        plt.clf()  # clear current figure
        plt.close("all")  # close all existing plots

        # Travis-CI does not have san-serif so ensure standard fonts are used.
        # Note that this must be set before each test otherwise it will be reset by
        # the Yellowbrick styles.
        mpl.rcParams["font.family"] = "DejaVu Sans"

    def assert_images_similar(
        self, visualizer=None, ax=None, tol=0.01, windows_tol=None, **kwargs
    ):
        """Accessible testing method for testing generation of a Visualizer.

        Requires the placement of a baseline image for comparison in the
        tests/baseline_images folder that corresponds to the module path of the
        VisualTestCase being evaluated. The name of the image corresponds to
        the test function where "self.assert_images_similar" is called.

        For example, calling "assert_images_similar" in the test function
        "test_class_report" in tests.test_classifier.test_class_balance would
        require placement a baseline image at:

        baseline_images/test_classifier/test_class_balance/test_class_report.png

        The easiest way to generate a baseline image is to first run the test that
        calls "assert_images_similar", and then copy the actual test generated
        image from:

        actual_images/

        visualizer : yellowbrick visualizer, default: None
            An instantiated yellowbrick visualizer that has been fitted,
            transformed and had all operations except for show called on it.

        ax : matplotlib Axes, default: None
            The axis to plot the figure on.

        tol : float, default: 0.01
            The tolerance (a color value difference, where 255 is the
            maximal difference).  The test fails if the average pixel
            difference is greater than this value.

        windows_tol: float, default: None
            Similar to the tol parameter, but targeted for testing on a
            windows environment.

        kwargs : dict
            Options to pass to the ImageComparison class.
        """
        # Hide this method from the pytest traceback on test failure.
        __tracebackhide__ = True

        # Build and execute the image comparison
        compare = ImageComparison(
            inspect.stack(),
            visualizer=visualizer,
            ax=ax,
            tol=tol,
            windows_tol=windows_tol,
            **kwargs
        )
        compare()

        # Return the compare object for meta testing
        return compare


##########################################################################
## Image Comparison Test
##########################################################################


class ImageComparison(object):
    """
    An image comparison wraps a single ``assert_images_similar`` statement to
    compose the actual and baseline image paths based on the stack the
    assertion was called in. It contains all properties that were formerly
    set on the test case (to facilitate our transition to pytest) so that they
    are immutable with respect to the single image comparison. By
    encapsulating these details, it is easier to debug image comparisons in
    meta tests.

    Parameters
    ----------
    stack : list of FrameInfo
        The list of the frame records for the caller's stack obtained from the
        ``inspect.stack()`` function. Must be called from the entry point to
        the image comparison (e.g. the next function from the test function).

    visualizer : Yellowbrick Visualizer instance, optional
        An instantiated Yellowbrick visualizer that wraps a matplotlib Axes
        and has been drawn on via the Yellowbrick API.

    ax : matplotlib Axes, optional
        A direct reference to a matplotlib Axes that has been drawn on.

    tol : float, default: 0.01
        The tolerance as a color value difference, where 255 is the maximal
        difference. The test fails if the average pixel difference is greater
        than this value.

    windows_tol : float, default: 0.01
        The tolerance (tol) parameter for the windows operating system environment.

    ext : string, default: ".png"
        The file extension to save the actual and baseline images as.

    remove_ticks : bool, default: True
        Remove the major and minor ticks on both the y and x axis to make the
        image comparison simpler (since different OS may have varying fonts or
        system level preferences).

    remove_title : bool, default: True
        Remove the title since different OS may have varying fonts.

    remove_labels : bool, default: True
        Remove the x and y labels since different OS may have varying fonts.

    remove_legend : bool, default: True
        Remove the legend since different OS may have varying fonts.

    Raises
    ------
    ValueError : at least one of visualizer or ax must be specified.
    """

    def __init__(
        self,
        stack,
        visualizer=None,
        ax=None,
        tol=0.01,
        windows_tol=0.01,
        ext=".png",
        remove_ticks=True,
        remove_title=True,
        remove_labels=True,
        remove_legend=True,
    ):

        # Ensure we have something to draw on
        if visualizer is None and ax is None:
            raise ValueError("at least one of visualizer or ax must be specified")

        # Save the ax being drawn on
        self.ax = ax or visualizer.ax

        # Parse the stack for the test and module name, element 0 should be
        # the assertion function (or whatever called this init), element 1
        # should be the test function and start with the test.
        frame = stack[1]

        # FrameInfo(frame, filename, lineno, function, code_context, index)
        self.test_func_name = frame[3]
        if not self.test_func_name.startswith("test"):
            raise ValueError("{} is not a test function".format(self.test_func_name))

        # Find the relative path to the Yellowbrick tests to compute the
        # module name for storing images in the actual and baseline dirs.
        root = os.path.commonprefix((TESTS, frame[1]))
        module_path = os.path.relpath(frame[1], root)
        self.test_module_path = os.path.splitext(module_path)[0]

        # Set the error tolerance depending on the os
        if os.name == "nt" and windows_tol is not None:
            self.tol = windows_tol
        else:
            self.tol = tol

        # Save other image comparison properties
        self.ext = ext
        self.remove_ticks = remove_ticks
        self.remove_title = remove_title
        self.remove_labels = remove_labels
        self.remove_legend = remove_legend

    def __call__(self):
        """
        Executes the image comparison by cleaning up the actual figure, saving
        the actual figure, then comparing the actual figure to the baseline.
        """
        # Hide this method from the pytest traceback on test failure.
        __tracebackhide__ = True

        self.cleanup()
        self.save()
        self.compare()

    @property
    def actual_image_path(self):
        """
        Computes the path in ACTUAL_IMAGES to the test image based on the test
        name and module. Creates any required parent dirs along the way.
        """
        return self._image_path(ACTUAL_IMAGES)

    @property
    def baseline_image_path(self):
        """
        Computes the path in BASELINE_IMAGES to the expected location of the
        file to compare the actual image against. Creates any required parent
        dirs along the way.
        """
        return self._image_path(BASELINE_IMAGES)

    def _image_path(self, root):
        """
        Computes the image path from the specified root directory (shared
        functionality for both actual and baseline image paths).
        """
        # Directory the images for this module are stored in.
        imgdir = os.path.join(root, self.test_module_path)

        # Create directory if it doesn't exist
        if not os.path.exists(imgdir):
            os.makedirs(imgdir, mode=0o777, exist_ok=True)

        # Create the image path from the test name
        return os.path.join(imgdir, self.test_func_name + self.ext)

    def cleanup(self):
        """
        Cleanup the image by removing textual/formatting elements.
        """
        # Hide this method from the pytest traceback on test failure.
        __tracebackhide__ = True

        if self.remove_title:
            self.ax.set_title("")

        if self.remove_ticks:
            null_formatter = ticker.NullFormatter()
            for axis in ("xaxis", "yaxis", "zaxis"):
                try:
                    axis = getattr(self.ax, axis)
                    axis.set_major_formatter(null_formatter)
                    axis.set_minor_formatter(null_formatter)
                except AttributeError:
                    continue

        if self.remove_labels:
            self.ax.set_xlabel("")
            self.ax.set_ylabel("")

        if self.remove_legend and self.ax.get_legend() is not None:
            self.ax.legend_.remove()

    def save(self):
        """
        Save the actual image to disk after cleaning it up.
        """
        # Hide this method from the pytest traceback on test failure.
        __tracebackhide__ = True

        assert self.ax.has_data(), "nothing has been drawn on the Axes"
        self.ax.get_figure().savefig(self.actual_image_path)

    def compare(self):
        """
        Compare the actual image to the baseline image, raising an exception
        if the baseline image does not exist or if the image is not close.
        """
        # Hide this method from the pytest traceback on test failure.
        __tracebackhide__ = True

        # Get expected and actual paths
        expected = self.baseline_image_path
        actual = self.actual_image_path

        # Assert that we have an actual image already saved
        assert os.path.exists(actual), "actual image hasn't been saved yet"

        # Ensure we have an image to compare against (common failure)
        if not os.path.exists(expected):
            raise ImageComparisonFailure(
                "baseline image does not exist:\n{}".format(os.path.relpath(expected))
            )
        # Perform the comparison
        err = compare_images(expected, actual, self.tol, in_decorator=True)

        # Raise image comparison failure if not close
        if err:
            for key in ("actual", "expected"):
                err[key] = os.path.relpath(err[key])

            raise ImageComparisonFailure(
                (
                    "images not close (RMS {rms:0.3f})" "\n{actual}\n\tvs\n{expected}"
                ).format(**err)
            )
