# tests.test_utils.test_target
# Tests for the target helper functions module.
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Thu Dec 27 20:43:31 2018 -0500
#
# For license information, see LICENSE.txt
#
# ID: test_target.py [899c88a] benjamin@bengfort.com $

"""
Tests for the target helper functions module.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np

from yellowbrick.utils.target import *
from yellowbrick.exceptions import YellowbrickValueError
from sklearn.datasets import make_regression, make_classification


##########################################################################
## Target Color Type Tests
##########################################################################


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, TargetType.SINGLE),
        (np.ones(15), TargetType.SINGLE),
        (["a", "b", "a", "b", "c"], TargetType.DISCRETE),
        ([1, 2, 1, 2, 3], TargetType.DISCRETE),
        ([0.23, 0.94, 1.3, -1.02, 0.11], TargetType.CONTINUOUS),
        ([1, 2, 0.2, 0.5, 1], TargetType.CONTINUOUS),
        (np.array([0.2, 2.2, 1.2, -3.1]), TargetType.CONTINUOUS),
        (np.array([[1, 2], [0, 2], [2, 1]]), TargetType.DISCRETE),
        (np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]]), TargetType.UNKNOWN),
    ],
    ids=[
        "none",
        "ones",
        "list str",
        "list int",
        "list float",
        "mixed list",
        "float array",
        "multioutput",
        "cube",
    ],
)
def test_target_color_type(value, expected):
    """
    Test the target_color_type helper function with a variety of data types
    """
    assert target_color_type(value) == expected


@pytest.mark.parametrize(
    "n_classes,expected",
    [
        (2, TargetType.DISCRETE),
        (4, TargetType.DISCRETE),
        (MAX_DISCRETE_CLASSES, TargetType.DISCRETE),
        (MAX_DISCRETE_CLASSES + 3, TargetType.CONTINUOUS),
    ],
    ids=["binary", "multiclass", "max discrete", "too many discrete"],
)
def test_binary_target_color_type(n_classes, expected):
    """
    Test classification target color type
    """
    _, y = make_classification(n_classes=n_classes, n_informative=n_classes + 2)
    assert target_color_type(y) == expected


def test_regression_target_color_type():
    """
    Test regression target color type
    """
    _, y = make_regression()
    assert target_color_type(y) == TargetType.CONTINUOUS


@pytest.mark.parametrize(
    "val",
    [
        "auto",
        "single",
        "discrete",
        "continuous",
        "unknown",
        TargetType.AUTO,
        TargetType.SINGLE,
        TargetType.DISCRETE,
        TargetType.CONTINUOUS,
        TargetType.UNKNOWN,
    ],
)
def test_target_type_validate_valid(val):
    try:
        TargetType.validate(val)
    except YellowbrickValueError:
        pyetst.fail("valid target type raised validation error")


@pytest.mark.parametrize(
    "val", ["foo", 1, 3.14, "s", "DISCRETE", "CONTINUOUS", ["a", "b", "c"]]
)
def test_target_type_validate_invalid(val):
    with pytest.raises(YellowbrickValueError, match="unknown target color type"):
        TargetType.validate(val)


@pytest.mark.parametrize(
    "val,expected",
    [
        ("discrete", True),
        (TargetType.DISCRETE, True),
        ("DISCRETE", True),
        (8, False),
        ("FOO", False),
        (3.14, False),
        ("foo", False),
        (["discrete"], False),
        ({"discrete"}, False),
    ],
)
def test_target_type_equals(val, expected):
    assert (TargetType.DISCRETE == val) is expected
