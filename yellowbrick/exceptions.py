# yellowbrick.exceptions
# Exceptions hierarchy for the yellowbrick library
#
# Author:   Benjamin Bengfort
# Created:  Fri Jun 03 10:39:41 2016 -0700
#
# Copyright (C) 2016 The sckit-yb developers
# For license information, see LICENSE.txt
#
# ID: exceptions.py [cb75e0e] benjamin@bengfort.com $

"""
Exceptions and warnings hierarchy for the yellowbrick library
"""

##########################################################################
## Exceptions Hierarchy
##########################################################################


class YellowbrickError(Exception):
    """
    The root exception for all yellowbrick related errors.
    """

    pass


class VisualError(YellowbrickError):
    """
    A problem when interacting with matplotlib or the display framework.
    """

    pass


class ModelError(YellowbrickError):
    """
    A problem when interacting with sklearn or the ML framework.
    """

    pass


class NotFitted(ModelError):
    """
    An action was called that requires a fitted model.
    """

    @classmethod
    def from_estimator(klass, estimator, method=None):
        method = method or "this method"
        message = (
            "this {} instance is not fitted yet, please call fit "
            "with the appropriate arguments before using {}"
        ).format(estimator.__class__.__name__, method)
        return klass(message)


class DatasetsError(YellowbrickError):
    """
    A problem occured when interacting with data sets.
    """

    pass


class YellowbrickTypeError(YellowbrickError, TypeError):
    """
    There was an unexpected type or none for a property or input.
    """

    pass


class YellowbrickValueError(YellowbrickError, ValueError):
    """
    A bad value was passed into a function.
    """

    pass


class YellowbrickKeyError(YellowbrickError, KeyError):
    """
    An invalid key was used in a hash (dict or set).
    """

    pass


class YellowbrickAttributeError(YellowbrickError, AttributeError):
    """
    A required attribute is missing on the estimator.
    """

    pass


##########################################################################
## Assertions
##########################################################################


class YellowbrickAssertionError(YellowbrickError, AssertionError):
    """
    Used to indicate test failures.
    """

    pass


class ImageComparisonFailure(YellowbrickAssertionError):
    """
    Provides a cleaner error when image comparison assertions fail.
    """

    pass


##########################################################################
## Warnings
##########################################################################


class YellowbrickWarning(UserWarning):
    """
    Warning class used to notify users of Yellowbrick-specific issues.
    """

    pass


class DataWarning(YellowbrickWarning):
    """
    The supplied data has an issue that may produce unexpected visualizations.
    """

    pass
