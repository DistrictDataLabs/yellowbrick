# yellowbrick.utils.decorators
# Decorators and descriptors for annotating yellowbrick library functions.
#
# Author:   Benjamin Bengfort
# Created:  Thu May 18 15:13:33 2017 -0400
#
# Copyright (C) 2017 The sckit-yb developers
# For license information, see LICENSE.txt
#
# ID: decorators.py [79cd8cf] benjamin@bengfort.com $

"""
Decorators and descriptors for annotating yellowbrick library functions.
"""

##########################################################################
## Imports
##########################################################################

from functools import wraps


##########################################################################
## Decorators
##########################################################################


def memoized(fget):
    """
    Return a property attribute for new-style classes that only calls its
    getter on the first access. The result is stored and on subsequent
    accesses is returned, preventing the need to call the getter any more.

    Parameters
    ----------
    fget: function
        The getter method to memoize for subsequent access.

    See also
    --------
    python-memoized-property
        `python-memoized-property <https://github.com/estebistec/python-memoized-property>`_
    """
    attr_name = "_{0}".format(fget.__name__)

    @wraps(fget)
    def fget_memoized(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fget(self))
        return getattr(self, attr_name)

    return property(fget_memoized)


class docutil(object):
    """
    This decorator can be used to apply the doc string from another function
    to the decorated function. This is used for our single call wrapper
    functions who implement the visualizer API without forcing the user to
    jump through all the hoops. The docstring of both the visualizer and the
    single call wrapper should be identical, this decorator ensures that we
    only have to edit one doc string.

    Usage::

        @docutil(Visualizer.__init__)
        def visualize(*args, **kwargs):
            pass

    The basic usage is that you instantiate the decorator with the function
    whose docstring you want to copy, then apply that decorator to the the
    function whose docstring you would like modified.

    Note that this decorator performs no wrapping of the target function.
    """

    def __init__(self, func):
        """Create a decorator to document other functions with the specified
        function's doc string.

        Parameters
        ----------
        func : function
            The function whose doc string we should decorate with
        """
        self.doc = func.__doc__

    def __call__(self, func):
        """Modify the decorated function with the stored doc string.

        Parameters
        ----------
        func: function
            The function to apply the saved doc string to.
        """
        func.__doc__ = self.doc
        return func
