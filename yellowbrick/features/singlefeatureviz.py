# yellowbrick.features.singlefeatureviz
# Implements a single feature visualizer.
#
# Author:   Liam Schumm <lschumm@protonmail.com>
# Created:  Mon May 06 11:30:00 2019 -0400
#
# Copyright (C) 2016-2019 District Data Labs
# For license information, see LICENSE.txt
#

##########################################################################
## Imports
##########################################################################

import numpy as np
from yellowbrick.features.base import DataVisualizer
from yellowbrick.exceptions import YellowbrickValueError

try:
    import pandas as pd
except ImportError:
    pd = None

##########################################################################
## SingleFeatureViz class definition
##########################################################################

class SingleFeatureViz(DataVisualizer):
    def __init__(self, idx, plot_type="violin", ax=None, features=None, classes=None, color=None,
                 colormap=None, **kwargs):
        """
        Initialize the data visualization with many of the options required
        in order to make most visualizations work.

        Parameters
        ----------
        idx : int or str
              The index of the feature to visualize

        plot_type : str
                    Defaults to "violin". Can be "violin", "box", or "scatter".
        """
        super(DataVisualizer, self).__init__(ax=ax, features=features, **kwargs)

        # Data Parameters
        self.features_  = features
        self.classes_  = classes
        self.idx = idx

        # Visual Parameters
        self.plot_type = plot_type
        self.color = color
        self.colormap = colormap

        if self.plot_type not in ["violin", "hist", "box"]:
            raise YellowBrickValueError("{plot_type} is not a valid plot_type for SingleFeatureViz".format(plot_type=x))

    
    def fit(self, X, y=None, **kwargs):
        """
        The fit method is the primary drawing input for the
        visualization since it has both the X and y data required for the
        viz and the transform method does not.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs : dict
            Pass generic arguments to the drawing method

        Returns
        -------
        self : instance
            Returns the instance of the transformer/visualizer
        """
        super(DataVisualizer, self).fit(X, y, **kwargs)

        if pd and isinstance(X, pd.DataFrame):
            if isinstance(self.idx, int):
                raise YellowBrickValueError("A string index is required for a Pandas DataFrame")
            x = X[self.idx]
            self.ax.set_xlabel(self.idx)
        else:
            if isinstance(self.idx, str):
                if features is None:
                    raise YellowBrickValueError("A string index is specified without a features list on a NumPy array")
                self.idx = features.index(self.idx)
            x = X[:,self.idx]
            if self.features_ is not None:
                self.ax.set_xlabel(self.features_[self.idx])

        if self.plot_type == 'hist':
            self.ax.hist(x)
            self.ax.set_ylabel('frequency', fontsize=16)

        elif self.plot_type == 'box':
            self.ax.boxplot([x])
                
        elif self.plot_type == 'violin':
            self.ax.violinplot(x)
                
        # Fit always returns self.
        return self
