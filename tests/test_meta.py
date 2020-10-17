# tests.test_meta
# Meta testing for testing helper functions!
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sat Apr 07 13:16:53 2018 -0400
#
# ID: test_meta.py [0a2d2b4] benjamin@bengfort.com $

"""
Meta testing for testing helper functions!
"""

##########################################################################
## Imports
##########################################################################

import os
import pytest
import inspect

import matplotlib as mpl

from tests.rand import RandomVisualizer
from unittest.mock import MagicMock, patch
from tests.base import ACTUAL_IMAGES, BASELINE_IMAGES
from tests.base import VisualTestCase, ImageComparison

from yellowbrick.exceptions import ImageComparisonFailure


def assert_path_exists(*parts):
    # Hide this method from the pytest traceback on test failure.
    __tracebackhide__ = True

    path = os.path.join(*parts)
    assert os.path.exists(path), "expected {} to exist".format(path)


def assert_path_not_exists(*parts):
    # Hide this method from the pytest traceback on test failure.
    __tracebackhide__ = True

    path = os.path.join(*parts)
    assert not os.path.exists(path), "expected {} not to exist".format(path)


##########################################################################
## Test Cases
##########################################################################


class TestMetaImageComparison(VisualTestCase):
    """
    Meta Test: ImageComparison test cases
    """

    def test_image_comparison(self):
        """
        Test the image comparison initialization and properties
        """

        def inner_assertion_function(ax):
            stack = inspect.stack()
            return ImageComparison(stack, ax=ax)

        ax = MagicMock()
        compare = inner_assertion_function(ax)
        assert compare.ax is ax
        assert compare.test_func_name == "test_image_comparison"
        assert compare.test_module_path == "test_meta"

        # Must use os.path.join for Windows/POSIX compatibility
        assert compare.actual_image_path.endswith(
            os.path.join(
                "tests", "actual_images", "test_meta", "test_image_comparison.png"
            )
        )
        assert compare.baseline_image_path.endswith(
            os.path.join(
                "tests", "baseline_images", "test_meta", "test_image_comparison.png"
            )
        )

    @patch.object(ImageComparison, "cleanup")
    @patch.object(ImageComparison, "save")
    @patch.object(ImageComparison, "compare")
    def test_image_comparison_call(self, mock_cleanup, mock_save, mock_compare):
        """
        Test that image comparison cleans up, saves, and compares
        """

        def inner_assertion_function():
            stack = inspect.stack()
            return ImageComparison(stack, ax=MagicMock())

        compare = inner_assertion_function()
        compare()

        mock_cleanup.assert_called_once()
        mock_save.assert_called_once()
        mock_compare.assert_called_once()

    def test_image_comparison_requires_ax(self):
        """
        Assert raises if an axes object is not supplied
        """
        with pytest.raises(ValueError, match="ax must be specified"):
            ImageComparison(inspect.stack())

    def test_image_comparison_not_in_assertion(self):
        """
        Assert raises when image comparison not in assertion helper
        """
        with pytest.raises(ValueError, match="not a test function"):
            stack = inspect.stack()
            ImageComparison(stack, ax=MagicMock())

    def test_missing_baseline_image(self):
        """
        Test that a missing basline image raises an exception
        """
        viz = RandomVisualizer(random_state=14).fit()
        viz.finalize()

        # Assert the baseline image does not exist
        assert_path_not_exists(
            BASELINE_IMAGES, "test_meta", "test_missing_baseline_image.png"
        )

        with pytest.raises(ImageComparisonFailure, match="image does not exist"):
            self.assert_images_similar(viz)

        # Assert the actual image was created (to copy to baseline)
        assert_path_exists(
            ACTUAL_IMAGES, "test_meta", "test_missing_baseline_image.png"
        )

    def test_random_visualizer(self):
        """
        Test that a random visualization is correctly compared to a baseline
        """
        viz = RandomVisualizer(random_state=111).fit()
        viz.finalize()

        assert mpl.get_backend() == "agg"

        compare = self.assert_images_similar(viz, tol=1.0)
        assert_path_exists(compare.actual_image_path)
        assert_path_exists(compare.baseline_image_path)

    def test_random_visualizer_not_close(self):
        """
        Test that not close visualizers raise an assertion error.
        """
        # Baseline image random_state=224
        # NOTE: if regenerating baseline images, skip this one or change random state!
        viz = RandomVisualizer(random_state=225).fit()
        viz.finalize()

        with pytest.raises(ImageComparisonFailure, match="images not close"):
            # If failing, perhaps baseline images were regenerated? See note above.
            self.assert_images_similar(viz)

        # Assert there is a diff
        assert_path_exists(
            ACTUAL_IMAGES,
            "test_meta",
            "test_random_visualizer_not_close-failed-diff.png",
        )

    def test_random_visualizer_increased_tolerance(self):
        """
        Test that not close visualizers pass with increased tolerance
        """
        viz = RandomVisualizer(random_state=224).fit()
        viz.finalize()

        self.assert_images_similar(viz, tol=30)
