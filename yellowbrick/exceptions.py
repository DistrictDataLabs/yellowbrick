# yellowbrick.exceptions
# Exceptions hierarchy for the yellowbrick library
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Jun 03 10:39:41 2016 -0700
#
# Copyright (C) 2016 District Data Labs
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
