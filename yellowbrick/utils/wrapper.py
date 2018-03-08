# yellowbrick.utils.wrapper
# Utility package that provides a wrapper for new style classes.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Sun May 21 20:27:32 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: wrapper.py [b2ecd50] benjamin@bengfort.com $

"""
Utility package that provides a wrapper for new style classes.
"""

##########################################################################
## Wrapper Class
##########################################################################


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
        # proxy to the wrapped object
        return getattr(self._wrapped, attr)
