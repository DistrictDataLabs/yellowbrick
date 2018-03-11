
##########################################################################
## Imports
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
import math

from yellowbrick.base import Visualizer
from yellowbrick.exceptions import YellowbrickValueError

class MultipleVisualizer(Visualizer):
    """
    Used as a base class for visualizers that use subplots.

    Parameters
    ----------
    visualizers : A list of instantiated visualizers

    nrows: integer, default: None
        The number of rows desired, if you would like a fixed number of rows.
        Specify only one of nrows and ncols, the other should be None. If you 
        specify nrows, there will be enough columns created to fit all the 
        visualizers specified in the visualizers list. 
    
    ncols: integer, default: None
        The number of columns desired, if you would like a fixed number of columns.
        Specify only one of nrows and ncols, the other should be None. If you 
        specify ncols, there will be enough rows created to fit all the 
        visualizers specified in the visualizers list. 

    axarr: matplotlib.axarr, default: None.
        If you want to put the plot onto an existing axarr, specify it here. Otherwise a new
        one will be created. 

    kwargs : additional keyword arguments, default: None
        Any additional keyword arguments will be passed on to the fit() method and therefore 
        passed on to the fit() method of the wrapped estimators, if applicable. Otherwise ignored.

    Examples
    --------

    >>> from sklearn.linear_model import LogisticRegression
    >>> from yellowbrick.classifier import ConfusionMatrix
    >>> from yellowbrick.classifier import ClassBalance
    >>> model = LogisticRegression()
    >>> visualizers = [ClassBalance(model),ConfusionMatrix(model)]
    >>> mv = MultipleVisualizer(visualizers, ncols=2)
    >>> mv.fit(X_train, y_train)
    >>> mv.score(X_test, y_test)
    >>> mv.poof()
    """
    def __init__(self, visualizers = [], nrows = None, ncols = None, axarr = None, **kwargs):
        self.visualizers = visualizers
        self.plotcount = len(visualizers)
        if nrows == None and ncols == None:
            self.ncols = 1
            self.nrows = self.plotcount
        elif ncols == None:
            self.nrows = nrows
            self.ncols = math.ceil(self.plotcount / self.nrows)
        elif nrows == None:
            self.ncols = ncols
            self.nrows = math.ceil(self.plotcount / self.ncols)
        else:
            raise YellowbrickValueError("You can only specify either nrows or ncols, \
                the other will be calculated based on the length of the list of visualizers.")
        

        if axarr == None:
            fig, axarr = plt.subplots(self.nrows, self.ncols, squeeze = False)
        
        self.axarr = axarr

        idx = 0
        for row in range(self.nrows):
            for col in range(self.ncols):
                try:
                    self.visualizers[idx].ax = self.axarr[row, col]
                #If len(visualizers) isn't evenly divisibly by rows/columns, 
                #we want to create the illusion of empty space by hiding the axis
                except IndexError as e:
                    self.axarr[row,col].axis('off')

                idx += 1

        self.kwargs = kwargs

    def fit(self,X,y):

        for idx in range(len(self.visualizers)):
            self.visualizers[idx].fit(X,y)

        return self

    def score(self,X,y):

        for idx in range(len(self.visualizers)):
            self.visualizers[idx].score(X,y)

        return self

    def poof(self, outpath=None, **kwargs):
        
        if self.axarr is None: return

        #Finalize all visualizers
        for idx in range(len(self.visualizers)):
            self.visualizers[idx].finalize()

        #Choose a reasonable default size if the user has not manually specified one
        # self.size() uses pixels rather than matplotlib's default of inches
        if not hasattr(self, "_size") or self._size is None:
            self._width = 400 * self.ncols
            self._height = 400 * self.nrows
            self.size = (self._width,self._height);

        if outpath is not None:
            plt.savefig(outpath, **kwargs)
        else:
            plt.show()

