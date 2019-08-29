# yellowbrick.utils.target
# Helper functions related to the target variable.
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Thu Dec 27 20:16:18 2018 -0500
#
# Copyright (C) 2018 The sckit-yb developers
# For license information, see LICENSE.txt
#
# ID: target.py [899c88a] benjamin@bengfort.com $

"""
Helper functions related to the target variable.
"""

##########################################################################
## Imports and Module Variables
##########################################################################

import numpy as np

from enum import Enum
from sklearn.utils.multiclass import type_of_target
from yellowbrick.exceptions import YellowbrickValueError


__all__ = ["MAX_DISCRETE_CLASSES", "TargetType", "target_color_type"]

MAX_DISCRETE_CLASSES = 12


class TargetType(Enum):
    """Constants for defining target colors by input type"""

    AUTO = "auto"
    SINGLE = "single"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    UNKNOWN = "unknown"

    @classmethod
    def validate(klass, val):
        if isinstance(val, klass):
            return

        try:
            klass(val)
        except ValueError:
            raise YellowbrickValueError("unknown target color type '{}'".format(val))

    def __eq__(self, other):
        if isinstance(other, str):
            try:
                return TargetType(other.lower()) == self
            except ValueError:
                return False
        return super(TargetType, self).__eq__(other)


##########################################################################
## Helper Functions
##########################################################################


def target_color_type(y):
    """
    Determines the type of color space that will best represent the target
    variable y, e.g. either a discrete (categorical) color space or a
    continuous color space that requires a colormap. This function can handle
    both 1D or column vectors as well as multi-output targets.

    Parameters
    ----------
    y : array-like
        Must be a valid array-like data structure that can be passed to a
        scikit-learn supervised estimator.

    Returns
    -------
    color_type : string
        One of:

        * 'discrete': `y` is either a binary target or a multiclass target
          with <= 12 discrete classes.
        * 'continuous': `y` is an array-like of floats that are not all
          integers or a multiclass target with > 12 discrete classes.
        * 'unknown': `y` is array-like but none of the above. For example
          a multilabel-indicator or a 3D array. No exception is raised.
    """
    if y is None or len(np.unique(y)) == 1:
        return TargetType.SINGLE

    ttype = type_of_target(y)

    if ttype.startswith("continuous"):
        return TargetType.CONTINUOUS

    if ttype.startswith("binary"):
        return TargetType.DISCRETE

    if ttype.startswith("multiclass"):
        if len(np.unique(y)) > MAX_DISCRETE_CLASSES:
            return TargetType.CONTINUOUS
        return TargetType.DISCRETE

    return TargetType.UNKNOWN
