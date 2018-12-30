# tests.test_utils.test_target
# Tests for the target helper functions module.
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Thu Dec 27 20:43:31 2018 -0500
#
# For license information, see LICENSE.txt
#
# ID: test_target.py [] benjamin@bengfort.com $

"""
Tests for the target helper functions module.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np

from yellowbrick.utils.target import *
from sklearn.datasets import make_regression, make_classification


##########################################################################
## Target Color Type Tests
##########################################################################

@pytest.mark.parametrize("value,expected", [
    (['a', 'b', 'a', 'b', 'c'], DISCRETE),
    ([1, 2, 1, 2, 3], DISCRETE),
    ([.23, 0.94, 1.3, -1.02, 0.11], CONTINUOUS),
    ([1, 2, 0.2, 0.5, 1], CONTINUOUS),
    (np.array([0.2, 2.2, 1.2, -3.1]), CONTINUOUS),
    (np.array([[1, 2], [0, 2], [2, 1]]), DISCRETE),
    (np.array([[[1,2], [1,2]], [[1,2], [1,2]]]), UNKNOWN),
], ids=['list str', 'list int', 'list float', 'mixed list', 'float array', 'multioutput', 'cube'])
def test_target_color_type(value, expected):
    """
    Test the target_color_type helper function with a variety of data types
    """
    assert target_color_type(value) == expected


@pytest.mark.parametrize("n_classes,expected", [
    (2, DISCRETE),
    (4, DISCRETE),
    (MAX_DISCRETE_CLASSES, DISCRETE),
    (MAX_DISCRETE_CLASSES+3, CONTINUOUS),
], ids=["binary", "multiclass", "max discrete", "too many discrete"])
def test_binary_target_color_type(n_classes, expected):
    """
    Test classification target color type
    """
    _, y = make_classification(n_classes=n_classes, n_informative=n_classes+2)
    assert target_color_type(y) == expected


def test_regression_target_color_type():
    """
    Test regression target color type
    """
    _, y = make_regression()
    assert target_color_type(y) == CONTINUOUS
