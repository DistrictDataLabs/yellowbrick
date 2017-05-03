
from .base import ClassificationScoreVisualizer
from ..utils import get_model_name
from ..style.palettes import LINE_COLOR

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import auc, roc_auc_score, roc_curve



##########################################################################
## Receiver Operating Characteristics
##########################################################################

class ROCAUC(ClassificationScoreVisualizer):
    """
    Plot the ROC to visualize the tradeoff between the classifier's
    sensitivity and specificity.

    Parameters
    ----------

    ax : the axis to plot the figure on.

    model : the Scikit-Learn estimator
        Should be an instance of a classifier, else the __init__ will
        return an error.

    roc_color : color of the ROC curve
        Specify the color as a matplotlib color: you can specify colors in
        many weird and wonderful ways, including full names ('green'), hex
        strings ('#008000'), RGB or RGBA tuples ((0,1,0,1)) or grayscale
        intensities as a string ('0.8').

    diagonal_color : color of the diagonal
        Specify the color as a matplotlib color.

    kwargs : keyword arguments passed to the super class.
        Currently passing in hard-coded colors for the Receiver Operating
        Characteristic curve and the diagonal.
        These will be refactored to a default Yellowbrick style.

    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.


    Examples
    --------

    >>> from yellowbrick.classifier import ROCAUC
    >>> from sklearn.linear_model import LogisticRegression
    >>> logistic = LogisticRegression()
    >>> viz = ROCAUC(logistic)
    >>> viz.fit(X_train, y_train)
    >>> viz.score(X_test, y_test)
    >>> viz.poof()
    
    """
    def __init__(self, model, ax=None, **kwargs):
        
        super(ROCAUC, self).__init__(model, ax=ax, **kwargs)

        ## hoisted to ScoreVisualizer base class
        self.name = get_model_name(self.estimator)

        # Color map defaults as follows:
        # ROC color is the current color in the cycle
        # Diagonal color is the default LINE_COLOR
        self.colors = {
            'roc': kwargs.pop('roc_color', None),
            'diagonal': kwargs.pop('diagonal_color', LINE_COLOR),
        }

    def score(self, X, y=None, **kwargs):
        """
        Generates the predicted target values using the Scikit-Learn
        estimator.

        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        Returns
        ------

        ax : the axis with the plotted figure

        """
        y_pred = self.predict(X)
        self.fpr, self.tpr, self.thresholds = roc_curve(y, y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        return self.draw(y, y_pred)

    def draw(self, y, y_pred):
        """
        Renders ROC-AUC plot.
        Called internally by score, possibly more than once

        Parameters
        ----------

        y : ndarray or Series of length n
            An array or series of target or class values

        y_pred : ndarray or Series of length n
            An array or series of predicted target values

        Returns
        ------

        ax : the axis with the plotted figure

        """
        # Create the axis if it doesn't exist
        if self.ax is None:
            self.ax = plt.gca()

        plt.plot(self.fpr, self.tpr, c=self.colors['roc'], label='AUC = {:0.2f}'.format(self.roc_auc))

        # Plot the line of no discrimination to compare the curve to.
        plt.plot([0,1],[0,1],'m--',c=self.colors['diagonal'])

        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Set the title and add the legend
        self.set_title('ROC for {}'.format(self.name))
        self.ax.legend(loc='lower right')

        # Set the limits for the ROC/AUC (always between 0 and 1)
        self.ax.set_xlim([-0.02, 1.0])
        self.ax.set_ylim([ 0.00, 1.1])


def roc_auc(model, X, y=None, ax=None, **kwargs):
    """Quick method:

    Displays the tradeoff between the classifier's
    sensitivity and specificity.

    This helper function is a quick wrapper to utilize the ROCAUC
    ScoreVisualizer for one-off analysis.

    Parameters
    ----------
    X  : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features.

    y  : ndarray or Series of length n
        An array or series of target or class values.

    ax : matplotlib axes
        The axes to plot the figure on.

    model : the Scikit-Learn estimator (should be a classifier)

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the roc-auc curve was drawn on.
    """
    # Instantiate the visualizer
    visualizer = ROCAUC(model, ax, **kwargs)

    # Create the train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X_train, y_train, **kwargs)
    visualizer.score(X_test, y_test)

    # Return the axes object on the visualizer
    return visualizer.ax
