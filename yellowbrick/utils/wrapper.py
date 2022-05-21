# yellowbrick.utils.wrapper
# Utility package that provides a wrapper for new style classes.
#
# Author:   Benjamin Bengfort
# Created:  Sun May 21 20:27:32 2017 -0700
#
# Copyright (C) 2017 The sckit-yb developers
# For license information, see LICENSE.txt
#
# ID: wrapper.py [b2ecd50] benjamin@bengfort.com $

"""
Utility package that provides a wrapper for new style classes.
"""

##########################################################################
## Wrapper Class
##########################################################################

from yellowbrick.exceptions import YellowbrickAttributeError, YellowbrickTypeError


class Wrapper(object):
    """
    Object wrapper class.

    An object wrapper is initialized with the object it wraps then proxies
    any unhandled getattribute methods to it. If no attribute is found either
    on the wrapper or the wrapped object, an AttributeError is raised.

    .. seealso:: http://code.activestate.com/recipes/577555-object-wrapper-class/

    Parameters
    ----------
    obj : object
        The object to wrap with the wrapper class
    """

    def __init__(self, obj):
        self._wrapped = obj

    def __getattr__(self, attr):
        if self is self._wrapped:
            raise YellowbrickTypeError("wrapper cannot wrap itself or recursion will occur")

        # proxy to the wrapped object
        try:
            return getattr(self._wrapped, attr)
        except AttributeError as e:
            raise YellowbrickAttributeError(f"neither visualizer '{self.__class__.__name__}' nor wrapped estimator '{type(self._wrapped).__name__}' have attribute '{attr}'") from e
