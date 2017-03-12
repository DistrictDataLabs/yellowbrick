# yellowbrick.neighbors.mixins
# Abstract neighbors mixins for Yellowbrick.
#
# Author:   Nathan Danielsen <rbilbro@gmail.com.com>
# Created:  Sat Mar 12 14:17:29 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt

from yellowbrick.exceptions import YellowbrickValueError

class BivariateFeatureMixin(object):
    """Mixin to ensure that only two features can be used"""

    def __init__(self, *args, **kwargs):
        """
        The features kwarg detemines if an exception is raised if passed in.

        Parameters
        ----------
        args: list, tuple
            Variable length argument list

        kwargs: dict
            keyword arguments.
        """
        features_ = kwargs.get('features', None)
        if features_ is not None:
            if len(features_) != 2:
                 raise YellowbrickValueError('{} only accepts two features'.format(self.__class__.__name__))
        super(BivariateFeatureMixin, self).__init__()
