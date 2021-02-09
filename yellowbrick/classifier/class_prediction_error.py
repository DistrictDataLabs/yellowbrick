# yellowbrick.classifier.class_prediction_error
# Shows the balance of classes and their associated predictions.
#
# Author:   Larry Gray
# Author:   Benjamin Bengfort
# Created:  Fri Jul 20 10:26:25 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: class_prediction_error.py [] lwgray@gmail.com $

"""
Shows the balance of classes and their associated predictions.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from sklearn.utils.multiclass import unique_labels

from yellowbrick.draw import bar_stack
from yellowbrick.classifier.base import ClassificationScoreVisualizer
from yellowbrick.exceptions import ModelError, YellowbrickValueError, NotFitted

try:
    # See #1124: this allows compatibility for scikit-learn >= 0.20
    from sklearn.metrics._classification import _check_targets
except ImportError:
    from sklearn.metrics.classification import _check_targets


##########################################################################
## Class Prediction Error Chart
##########################################################################


class ClassPredictionError(ClassificationScoreVisualizer):
    """
    Class Prediction Error chart that shows the support for each class in the
    fitted classification model displayed as a stacked bar. Each bar is segmented
    to show the distribution of predicted classes for each class. It is initialized
    with a fitted model and generates a class prediction error chart on draw.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    classes : list of str, defult: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.

    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    force_model : bool, default: False
        Do not check to ensure that the underlying estimator is a classifier. This
        will prevent an exception when the visualizer is initialized but may result
        in unexpected or unintended behavior.

    kwargs : dict
        Keyword arguments passed to the visualizer base classes.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels observed while fitting.

    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting.

    score_ : float
        An evaluation metric of the classifier on test data produced when
        ``score()`` is called. This metric is between 0 and 1 -- higher scores are
        generally better. For classifiers, this score is usually accuracy, but
        ensure you check the underlying model for more details about the score.

    predictions_ : ndarray
        An ndarray of predictions whose rows are the true classes and
        whose columns are the predicted classes
    """

    def __init__(
        self,
        estimator,
        ax=None,
        classes=None,
        encoder=None,
        is_fitted="auto",
        force_model=False,
        **kwargs
    ):
        super(ClassPredictionError, self).__init__(
            estimator,
            ax=ax,
            classes=classes,
            encoder=encoder,
            is_fitted=is_fitted,
            force_model=force_model,
            **kwargs
        )

    def score(self, X, y):
        """
        Generates a 2D array where each row is the count of the
        predicted classes and each column is the true class

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        Returns
        -------
        score_ : float
            Global accuracy score
        """
        # Must be computed before calling super
        # We're relying on predict to raise NotFitted
        y_pred = self.predict(X)
        y_type, y_true, y_pred = _check_targets(y, y_pred)
        if y_type not in ("binary", "multiclass"):
            raise YellowbrickValueError("{} is not supported".format(y_type))

        # Get the indices of the unique labels
        indices = unique_labels(y_true, y_pred)
        labels = self._labels()

        # Call super to compute self.score_ and verify classes
        try:
            super(ClassPredictionError, self).score(X, y)
        except ModelError as e:
            # raise visualizer-specific errors
            if labels is not None and len(labels) < len(indices):
                raise NotImplementedError(
                    "filtering classes is currently not supported"
                )
            else:
                raise e

        # Ensure all labels are used
        if labels is not None and len(labels) > len(indices):
            raise ModelError(
                "y and y_pred contain zero values for one of the specified classes"
            )

        # Create a table of predictions whose rows are the true classes
        # and whose columns are the predicted classes; each element
        # is the count of predictions for that class that match the true
        # value of that class.
        self.predictions_ = np.array(
            [
                [(y_pred[y == label_t] == label_p).sum() for label_p in indices]
                for label_t in indices
            ]
        )

        self.draw()
        return self.score_

    def draw(self):
        """
        Renders the class prediction error across the axis.

        Returns
        -------
        ax : Matplotlib Axes
            The axes on which the figure is plotted
        """

        if not hasattr(self, "predictions_") or not hasattr(self, "classes_"):
            raise NotFitted.from_estimator(self, "draw")

        legend_kws = {"bbox_to_anchor": (1.04, 0.5), "loc": "center left"}
        bar_stack(
            self.predictions_,
            self.ax,
            labels=list(self.classes_),
            ticks=self.classes_,
            colors=self.class_colors_,
            legend_kws=legend_kws,
        )
        return self.ax

    def finalize(self, **kwargs):
        """
        Adds a title and axis labels to the visualizer, ensuring that the
        y limits zoom the visualization in to the area of interest. Finalize
        also calls tight layout to ensure that no parts of the figure are
        cut off.

        Notes
        -----
        Generally this method is called from show and not directly by the user.
        """

        # Set the title
        self.set_title("Class Prediction Error for {}".format(self.name))

        # Set the axes labels
        self.ax.set_xlabel("actual class")
        self.ax.set_ylabel("number of predicted class")

        # Compute the ceiling for the y limit
        cmax = max([sum(predictions) for predictions in self.predictions_])
        self.ax.set_ylim(0, cmax + cmax * 0.1)

        # Ensure the legend fits on the figure
        self.fig.tight_layout(rect=[0, 0, 0.90, 1])


##########################################################################
## Quick Method
##########################################################################


def class_prediction_error(
    estimator,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    ax=None,
    classes=None,
    encoder=None,
    is_fitted="auto",
    force_model=False,
    show=True,
    **kwargs
):
    """Class Prediction Error

    Divides the dataset X and y into train and test splits, fits the model on the train
    split, then scores the model on the test split. The visualizer displays the support
    for each class in the fitted classification model displayed as a stacked bar plot.
    Each bar is segmented to show the distribution of predicted classes for each class.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.

    X_train : ndarray or DataFrame of shape n x m
        A feature array of n instances with m features the model is trained on.
        Used to fit the visualizer and also to score the visualizer if test splits are
        not directly specified.

    y_train : ndarray or Series of length n
        An array or series of target or class values. Used to fit the visualizer and
        also to score the visualizer if test splits are not specified.

    X_test : ndarray or DataFrame of shape n x m, default: None
        An optional feature array of n instances with m features that the model
        is scored on if specified, using X_train as the training data.

    y_test : ndarray or Series of length n, default: None
        An optional array or series of target or class values that serve as actual
        labels for X_test for scoring purposes.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    classes : list of str, defult: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.

    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    force_model : bool, default: False
        Do not check to ensure that the underlying estimator is a classifier. This
        will prevent an exception when the visualizer is initialized but may result
        in unexpected or unintended behavior.

    show: bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however
        you cannot call ``plt.savefig`` from this signature, nor
        ``clear_figure``. If False, simply calls ``finalize()``

    kwargs: dict
        Keyword arguments passed to the visualizer base classes.

    Returns
    -------
    viz : ClassPredictionError
        Returns the fitted, finalized visualizer
    """
    # Instantiate the visualizer
    viz = ClassPredictionError(
        estimator=estimator,
        ax=ax,
        classes=classes,
        encoder=encoder,
        is_fitted=is_fitted,
        force_model=force_model,
        **kwargs
    )

    # Fit the visualizer (calls draw)
    viz.fit(X_train, y_train, **kwargs)

    # Score the visualizer
    if X_test is not None and y_test is not None:
        viz.score(X_test, y_test)
    elif X_test is not None or y_test is not None:
        raise YellowbrickValueError("must specify both X_test and y_test or neither")
    else:
        viz.score(X_train, y_train)

    # Draw the final visualization
    if show:
        viz.show()
    else:
        viz.finalize()

    # Return the visualizer
    return viz
