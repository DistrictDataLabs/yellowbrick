# yellowbrick.utils.target
# Helper functions related to the target variable.
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Thu Dec 27 20:16:18 2018 -0500
#
# For license information, see LICENSE.txt
#
# ID: target.py [] benjamin@bengfort.com $

"""
Helper functions related to the target variable.
"""

##########################################################################
## Imports and Module Variables
##########################################################################

import numpy as np

from sklearn.utils.multiclass import type_of_target


__all__ = [
    'CONTINUOUS', 'DISCRETE', 'UNKNOWN', 'MAX_DISCRETE_CLASSES', 'target_color_type'
]

CONTINUOUS = "continuous"
DISCRETE   = "discrete"
UNKNOWN    = "unknown"
MAX_DISCRETE_CLASSES = 12


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
          integers or a multiclass with > 12 discrete classes.
        * 'unknown': `y` is array-like but none of the above. For example
          a multilabel-indeicator or a 3D array. No exception is raised.
    """
    ttype = type_of_target(y)

    if ttype.startswith(CONTINUOUS):
        return CONTINUOUS

    if ttype.startswith("binary"):
        return DISCRETE

    if ttype.startswith("multiclass"):
        if len(np.unique(y)) > MAX_DISCRETE_CLASSES:
            return CONTINUOUS
        return DISCRETE

    return UNKNOWN
