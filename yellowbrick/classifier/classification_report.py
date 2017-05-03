from .base import ClassificationScoreVisualizer


import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from ..utils import get_model_name
from ..style.palettes import color_sequence
from ..style import find_text_color

##########################################################################
## Classification Report
##########################################################################

class ClassificationReport(ClassificationScoreVisualizer):
    """
    Classification report that shows the precision, recall, and F1 scores
    for the model. Integrates numerical scores as well as a color-coded heatmap.
    
    Parameters
    ----------

    ax : The axis to plot the figure on.

    model : the Scikit-Learn estimator
        Should be an instance of a classifier, else the __init__ will
        return an error.

    classes : a list of class names for the legend
        If classes is None and a y value is passed to fit then the classes
        are selected from the target vector.

    colormap : optional string or matplotlib cmap to colorize lines
        Use sequential heatmap.

    kwargs : keyword arguments passed to the super class.
    
    Examples
    --------

    >>> from yellowbrick.classifier import ClassificationReport
    >>> from sklearn.linear_model import LogisticRegression
    >>> viz = ClassificationReport(LogisticRegression())
    >>> viz.fit(X_train, y_train)
    >>> viz.score(X_test, y_test)
    >>> viz.poof()

    """
    def __init__(self, model, ax=None, classes=None, **kwargs):

        super(ClassificationReport, self).__init__(model, ax=ax, **kwargs)

        ## hoisted to ScoreVisualizer base class
        self.estimator = model
        self.name = get_model_name(self.estimator)

        self.cmap = color_sequence(kwargs.pop('cmap', 'YlOrRd'))
        self.classes_ = classes

    def fit(self, X, y=None, **kwargs):
        """
        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: keyword arguments passed to Scikit-Learn API.
        """
        super(ClassificationReport, self).fit(X, y, **kwargs)
        if self.classes_ is None:
            self.classes_ = self.estimator.classes_
        return self

    def score(self, X, y=None, **kwargs):
        """
        Generates the Scikit-Learn classification_report

        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        """
        y_pred = self.predict(X)
        keys   = ('precision', 'recall', 'f1')
        self.scores = precision_recall_fscore_support(y, y_pred)
        self.scores = map(lambda s: dict(zip(self.classes_, s)), self.scores[0:3])
        self.scores = dict(zip(keys, self.scores))
        return self.draw(y, y_pred)

    def draw(self, y, y_pred):
        """
        Renders the classification report across each axis.

        Parameters
        ----------

        y : ndarray or Series of length n
            An array or series of target or class values

        y_pred : ndarray or Series of length n
            An array or series of predicted target values
        """
        # Create the axis if it doesn't exist
        if self.ax is None:
            self.ax = plt.gca()

        self.matrix = []
        for cls in self.classes_:
            self.matrix.append([self.scores['precision'][cls],self.scores['recall'][cls],self.scores['f1'][cls]])

        for column in range(0,3): #3 columns - prec,rec,f1
            for row in range(len(self.classes_)):
                current_score = self.matrix[row][column]
                base_color = self.cmap(current_score)
                text_color= find_text_color(base_color)

                self.ax.text(column,row,current_score,va='center',ha='center', color=text_color)

        fig = plt.imshow(self.matrix, interpolation='nearest', cmap=self.cmap, vmin=0, vmax=1,aspect='auto')

        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Set the title of the classifiation report
        self.set_title('{} Classification Report'.format(self.name))

        # Add the color bar
        plt.colorbar()

        # Compute the tick marks for both x and y
        x_tick_marks = np.arange(len(self.classes_)+1)
        y_tick_marks = np.arange(len(self.classes_))

        # Set the tick marks appropriately
        self.ax.set_xticks(x_tick_marks)
        self.ax.set_yticks(y_tick_marks)

        self.ax.set_xticklabels(['precision', 'recall', 'f1-score'], rotation=45)
        self.ax.set_yticklabels(self.classes_)

        # Set the labels for the two axes
        self.ax.set_ylabel('Classes')
        self.ax.set_xlabel('Measures')


def classification_report(model, X, y=None, ax=None, classes=None, **kwargs):
    """Quick method:

    Displays precision, recall, and F1 scores for the model.
    Integrates numerical scores as well color-coded heatmap.

    This helper function is a quick wrapper to utilize the ClassificationReport
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

    classes : list of strings
        The names of the classes in the target

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the classification report was drawn on.
    """
    # Instantiate the visualizer
    visualizer = ClassificationReport(model, ax, classes, **kwargs)

    # Create the train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X_train, y_train, **kwargs)
    visualizer.score(X_test, y_test)

    # Return the axes object on the visualizer
    return visualizer.ax