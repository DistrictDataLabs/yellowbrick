# yellowbrick.classifier
# Visualizations related to evaluating Scikit-Learn classification models
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Wed May 18 12:39:40 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: classifier.py [5eee25b] benjamin@bengfort.com $

"""
Visualizations related to evaluating Scikit-Learn classification models
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from .exceptions import YellowbrickTypeError
from .utils import get_model_name, isestimator, isclassifier
from .base import Visualizer, ScoreVisualizer, MultiModelMixin
from .style.palettes import color_sequence, color_palette, LINE_COLOR
from .style import find_text_color
from .utils import div_safe


##########################################################################
## Classification Visualization Base Object
##########################################################################

class ClassificationScoreVisualizer(ScoreVisualizer):

    """
    Base class for all ScoreVisualizers that evaluate a classification estimator.

    The primary functionality of this class is to perform a check to ensure
    the passed in estimator is a classifier, otherwise it raises a
    ``YellowbrickTypeError``.
    """

    def __init__(self, model, ax=None, **kwargs):
        # Check to see if model is an instance of a classifier.
        # Should return an error if it isn't.
        if not isclassifier(model):
            raise YellowbrickTypeError(
                "This estimator is not a classifier; try a regression or clustering score visualizer instead!"
        )

        super(ClassificationScoreVisualizer, self).__init__(model, ax=ax, **kwargs)

    #TODO during refactoring this can be used to generalize ClassBalance
    def class_counts(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

##########################################################################
## ConfusionMatrix
##########################################################################

class ConfusionMatrix(ClassificationScoreVisualizer):
    """
    Creates a heatmap visualization of the sklearn.metrics.confusion_matrix().
    Initialization: Requires a classification model

    Parameters
    ----------

    model : a Scikit-Learn classifier
        Should be an instance of a classifier otherwise a will raise a 
        YellowbrickTypeError exception on instantiation.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    classes : list, default: None
        a list of class names for the legend
        If classes is None and a y value is passed to fit then the classes
        are selected from the target vector.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------

    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression()
    >>> cm = ConfusionMatrix(model)
    >>> cm.fit(X_train, y_train)
    >>> cm.score(X_test, y_test)
    >>> cm.poof()

    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    def __init__(self, model, ax=None, classes=None, **kwargs):
        super(ConfusionMatrix, self).__init__(model, ax=ax, classes=None,**kwargs)
        #Parameters provided by super (for reference during development only):
        #self.ax
        #self.size
        #self.color
        #self.title
        #self.estimator
        #self.name

        #Initialize all the other attributes we'll use (for coder clarity)
        self.confusion_matrix = None

        self.cmap = color_sequence(kwargs.pop('cmap', 'YlOrRd'))
        self.cmap.set_under(color='w')
        self.cmap.set_over(color='#2a7d4f')
        self.edgecolors=[] #used to draw diagonal line for predicted class = true class

        
        #Convert list to array if necessary, since estimator.classes_ returns nparray
        self._classes = None if classes == None else np.array(classes)

    #TODO hoist this to shared confusion matrix / classification report heatmap class
    @property
    def classes(self):
        """
        Returns a numpy array of the classes in y
        Matches the user provided list if provided by the user in __init__
        If no list provided, tries to obtain it from the fitted estimator
        """
        if self._classes is None:
            try:
                print("trying")
                return self.estimator.classes_
            except AttributeError:
                return None
        return self._classes

    @classes.setter
    def classes(self, value):
        self._classes = value

    #todo hoist
    def fit(self, X, y=None, **kwargs):
        """
        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            keyword arguments passed to Scikit-Learn API.
        """
        super(ConfusionMatrix, self).fit(X, y, **kwargs)
        if self._classes is None:
            self.classes = self.estimator.classes_
        return self

    def score(self, X, y, sample_weight=None, percent=True):
        """
        Generates the Scikit-Learn confusion_matrix and applies this to the appropriate axis

        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        sample_weight: float, default: None
            optional, passed to the confusion_matrix

        percent: bool, default: True 
            Determines whether or not the confusion_matrix
            should be displayed as raw numbers or as a percent of the true
            predictions. Note, if using a subset of classes in __init__, percent should
            be set to False or inaccurate percents will be displayed.
        """
        y_pred = self.predict(X)

        self.confusion_matrix = confusion_matrix(y_true = y, y_pred = y_pred, labels=self.classes, sample_weight=sample_weight)
        self._class_counts = self.class_counts(y)
        
        #Make array of only the classes actually being used. 
        #Needed because sklearn confusion_matrix only returns counts for selected classes
            #but percent should be calculated based on all classes
        selected_class_counts = []
        for c in self.classes:
            try:
                selected_class_counts.append(self._class_counts[c])
            except KeyError:
                selected_class_counts.append(0)
        self.selected_class_counts = np.array(selected_class_counts)

        return self.draw(percent)

    def draw(self, percent=True):
        """
        Renders the classification report
        Should only be called internally, as it uses values calculated in Score
        and score calls this method.

        Parameters
        ----------

        percent:    bool
            Whether the heatmap should represent "% of True" or raw counts

        """
        # Create the axis if it doesn't exist
        if self.ax is None:
            self.ax = plt.gca()

        if percent == True:
            #Convert confusion matrix to percent of each row, i.e. the predicted as a percent of true in each class
            #div_safe function returns 0 instead of NAN.
            self._confusion_matrix_display = div_safe(
                    self.confusion_matrix,
                    self.selected_class_counts
                    )
            self._confusion_matrix_display =np.round(self._confusion_matrix_display* 100, decimals=0)
        else:
            self._confusion_matrix_display = self.confusion_matrix

        #Y axis should be sorted top to bottom in pcolormesh
        self._confusion_matrix_plottable = self._confusion_matrix_display[::-1,::]

        self.max = self._confusion_matrix_plottable.max()

        #Set up the dimensions of the pcolormesh
        X = np.linspace(start=0, stop=len(self.classes), num=len(self.classes)+1)
        Y = np.linspace(start=0, stop=len(self.classes), num=len(self.classes)+1)
        self.ax.set_ylim(bottom=0, top=self._confusion_matrix_plottable.shape[0])
        self.ax.set_xlim(left=0, right=self._confusion_matrix_plottable.shape[1])

        #Put in custom axis labels
        self.xticklabels = self.classes
        self.yticklabels = self.classes[::-1]
        self.xticks = np.arange(0, len(self.classes), 1) + .5
        self.yticks = np.arange(0, len(self.classes), 1) + .5
        self.ax.set(xticks=self.xticks, yticks=self.yticks)
        self.ax.set_xticklabels(self.xticklabels, rotation="vertical", fontsize=8)
        self.ax.set_yticklabels(self.yticklabels, fontsize=8)

        ######################
        # Add the data labels to each square
        ######################
        for x_index, x in np.ndenumerate(X):
            #np.ndenumerate returns a tuple for the index, must access first element using [0]
            x_index = x_index[0]
            for y_index, y in np.ndenumerate(Y):
                #Clean up our iterators
                #numpy doesn't like non integers as indexes; also np.ndenumerate returns tuple
                x_int = int(x)
                y_int = int(y)
                y_index = y_index[0]

                #X and Y are one element longer than the confusion_matrix. Don't want to add text for the last X or Y
                if x_index == X[-1] or y_index == Y[-1]:
                    break

                #center the text in the middle of the block
                text_x = x + 0.5
                text_y = y + 0.5

                #extract the value
                grid_val = self._confusion_matrix_plottable[x_int,y_int]
                
                #Determine text color
                scaled_grid_val = grid_val / self.max
                base_color = self.cmap(scaled_grid_val)
                text_color= find_text_color(base_color)

                #make zero values more subtle
                if self._confusion_matrix_plottable[x_int,y_int] == 0:
                    text_color = "0.75"

                #Put the data labels in the middle of the heatmap square
                self.ax.text(text_y,
                            text_x,
                            "{:.0f}{}".format(grid_val,"%" if percent==True else ""),
                            va='center',
                            ha='center',
                            fontsize=8,
                            color=text_color)

                #If the prediction is correct, put a bounding box around that square to better highlight it to the user
                #This will be used in ax.pcolormesh, setting now since we're iterating over the matrix
                    #ticklabels are conveniently already reversed properly to match the _confusion_matrix_plottalbe order
                if self.xticklabels[x_int] == self.yticklabels[y_int]:
                    self.edgecolors.append('black')
                else:
                    self.edgecolors.append('w')

        # Draw the heatmap. vmin and vmax operate in tandem with the cmap.set_under and cmap.set_over to alter the color of 0 and 100
        highest_count = self._confusion_matrix_plottable.max()
        vmax = 99.999 if percent == True else highest_count
        mesh = self.ax.pcolormesh(X,
                                  Y,
                                  self._confusion_matrix_plottable,
                                  vmin=0.00001,
                                  vmax=vmax,
                                  edgecolor=self.edgecolors,
                                  cmap=self.cmap,
                                  linewidth='0.01') #edgecolor='0.75', linewidth='0.01'
        return self.ax

    def finalize(self, **kwargs):
        self.set_title('{} Confusion Matrix'.format(self.name))
        self.ax.set_ylabel('True Class')
        self.ax.set_xlabel('Predicted Class')

##########################################################################
## Classification Report
##########################################################################

class ClassificationReport(ClassificationScoreVisualizer):
    """
    Classification report that shows the precision, recall, and F1 scores
    for the model. Integrates numerical scores as well color-coded heatmap.

    Parameters
    ----------

    model : a Scikit-Learn classifier
        Should be an instance of a classifier otherwise a will raise a 
        YellowbrickTypeError exception on instantiation.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    classes : list, default: None
        a list of class names for the legend
        If classes is None and a y value is passed to fit then the classes
        are selected from the target vector.

    colormap : string or cmap, default: None
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------

    >>> from sklearn.naive_bayes import GaussianNB
    >>> model = GaussianNB()
    >>> cm = ClassificationReport(model)
    >>> cm.fit(X_train, y_train)
    >>> cm.score(X_test, y_test)
    >>> cm.poof()

    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
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

        kwargs: dict
            keyword arguments passed to Scikit-Learn API.
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

        for column in range(len(self.matrix)+1):
            for row in range(len(self.classes_)):
                self.ax.text(column,row,self.matrix[row][column],va='center',ha='center')

        fig = plt.imshow(self.matrix, interpolation='nearest', cmap=self.cmap, vmin=0, vmax=1)

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


##########################################################################
## Quick Methods
##########################################################################

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

    ax : matplotlib Axes, default: None
        The axes to plot the figure on.

    model : a Scikit-Learn classifier
        Should be an instance of a classifier otherwise a will raise a 
        YellowbrickTypeError exception on instantiation.

    classes : list of strings, default: None
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

##########################################################################
## Receiver Operating Characteristics
##########################################################################

class ROCAUC(ClassificationScoreVisualizer):
    """
    Plot the ROC to visualize the tradeoff between the classifier's
    sensitivity and specificity.

    Parameters
    ----------

    model : a Scikit-Learn classifier
        Should be an instance of a classifier otherwise a will raise a 
        YellowbrickTypeError exception on instantiation.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    roc_color : color
        color of the ROC curve
        Specify the color as a matplotlib color: you can specify colors in
        many weird and wonderful ways, including full names ('green'), hex
        strings ('#008000'), RGB or RGBA tuples ((0,1,0,1)) or grayscale
        intensities as a string ('0.8').

    diagonal_color : color
        color of the diagonal
        Specify the color as a matplotlib color.

    classes : list, default: None
        a list of class names for the legend
        If classes is None and a y value is passed to fit then the classes
        are selected from the target vector.

    colormap : string or cmap, default: None
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.
        Currently passing in hard-coded colors for the Receiver Operating
        Characteristic curve and the diagonal.
        These will be refactored to a default Yellowbrick style.

    Examples
    --------

    >>> from sklearn.naive_bayes import GaussianNB
    >>> model = GaussianNB()
    >>> cm = ClassificationReport(model)
    >>> cm.fit(X_train, y_train)
    >>> cm.score(X_test, y_test)
    >>> cm.poof()

    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
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


##########################################################################
## Quick Methods
##########################################################################

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

    ax : matplotlib Axes, default: None
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


##########################################################################
## Class Balance Chart
##########################################################################

class ClassBalance(ClassificationScoreVisualizer):
    """
    Class balance chart that shows the support for each class in the
    fitted classification model displayed as a bar plot. It is initialized
    with a fitted model and generates a class balance chart on draw.

    Parameters
    ----------

    model : a Scikit-Learn classifier
        Should be an instance of a classifier otherwise a will raise a 
        YellowbrickTypeError exception on instantiation.

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    classes : list, default: None
        a list of class names for the legend
        If classes is None and a y value is passed to fit then the classes
        are selected from the target vector.

    kwargs: dict
        Keyword arguments passed to the super class. Here, used
        to colorize the bars in the histogram.

    Examples
    --------

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier()
    >>> visualizer = ClassBalance(model)
    >>> visualizer.fit(X_train, y_train)
    >>> visualizer.score(X_test, y_test)
    >>> visuazlier.poof()

    Notes
    -----

    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """
    def __init__(self, model, ax=None, classes=None, **kwargs):

        super(ClassBalance, self).__init__(model, ax=ax, **kwargs)

        self.colors    = color_palette(kwargs.pop('colors', None))
        self.classes_  = classes

    def fit(self, X, y=None, **kwargs):
        """
        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: keyword arguments passed to Scikit-Learn API.

        Returns
        -------
        self : instance
            Returns the instance of the classification score visualizer

        """
        super(ClassBalance, self).fit(X, y, **kwargs)
        if self.classes_ is None:
            self.classes_ = self.estimator.classes_
        return self

    def score(self, X, y=None, **kwargs):
        """
        Generates the Scikit-Learn precision_recall_fscore_support

        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        Returns
        -------

        ax : the axis with the plotted figure
        """
        y_pred = self.predict(X)
        self.scores  = precision_recall_fscore_support(y, y_pred)
        self.support = dict(zip(self.classes_, self.scores[-1]))
        return self.draw()

    def draw(self):
        """
        Renders the class balance chart across the axis.

        Returns
        -------

        ax : the axis with the plotted figure

        """
        # Create the axis if it doesn't exist
        if self.ax is None:
            self.ax = plt.gca()

        #TODO: Would rather not have to set the colors with this method.
        # Refactor to make better use of yb_palettes module?

        colors = self.colors[0:len(self.classes_)]
        plt.bar(np.arange(len(self.support)), self.support.values(), color=colors, align='center', width=0.5)

        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """

        # Set the title
        self.set_title('Class Balance for {}'.format(self.name))

        # Set the x ticks with the class names
        # TODO: change to the self.ax method rather than plt.xticks
        plt.xticks(np.arange(len(self.support)), self.support.keys())

        # Compute the ceiling for the y limit
        cmax, cmin = max(self.support.values()), min(self.support.values())
        self.ax.set_ylim(0, cmax + cmax* 0.1)

##########################################################################
## Quick Methods
##########################################################################

def class_balance(model, X, y=None, ax=None, classes=None, **kwargs):
    """Quick method:

    Displays the support for each class in the
    fitted classification model displayed as a bar plot.

    This helper function is a quick wrapper to utilize the ClassBalance
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
        Returns the axes that the class balance plot was drawn on.
    """
    # Instantiate the visualizer
    visualizer = ClassBalance(model, ax, classes, **kwargs)

    # Create the train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X_train, y_train, **kwargs)
    visualizer.score(X_test, y_test)

    # Return the axes object on the visualizer
    return visualizer.ax
