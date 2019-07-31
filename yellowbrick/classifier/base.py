# yellowbrick.classifier.base
# API for classification visualizer hierarchy.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Author:   Neal Humphrey
# Created:  Wed May 18 12:39:40 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: base.py [5388065] neal@nhumphrey.com $

"""
API for classification visualizer hierarchy.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from ..utils import isclassifier
from ..base import ScoreVisualizer
from ..exceptions import YellowbrickTypeError, YellowbrickValueError


##########################################################################
## Base Classification Visualizer
##########################################################################


class ClassificationScoreVisualizer(ScoreVisualizer):
    """Base class for classifier model selection.

    The ClassificationScoreVisualizer wraps a classifier to produce a
    visualization when the score method is called, usually to allow the user
    to effectively compare the performance between models.

    The base class provides helper functionality to ensure that classification
    visualizers are able to correctly identify and encode classes with human
    readable labels and to map colors to the classes if required.

    Parameters
    ----------
    model : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised.

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    classes : list of str, defult: None
        The class labels to use for the legend ordered by the index of the
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

    force_model : bool, default: False
        Do not check to ensure that the underlying estimator is a classifier. This
        will prevent an exception when the visualizer is initialized but may result
        in unexpected or unintended behavior.

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

    Notes
    -----
    Classification score visualizers should implement the ``score()``
    and ``draw()`` visualier methods.
    """

    def __init__(
        self,
        model,
        ax=None,
        fig=None,
        classes=None,
        encoder=None,
        force_model=False,
        **kwargs
    ):
        # A bit of type checking
        if not force_model and not isclassifier(model):
            raise YellowbrickTypeError(
                "This estimator is not a classifier; "
                "try a regression or clustering score visualizer instead!"
            )

        # Initialize the super method.
        super(ClassificationScoreVisualizer, self).__init__(
            model, ax=ax, fig=fig, **kwargs
        )

        self.set_params(
            classes=classes,
            encoder=encoder,
            force_model=force_model,
        )

    def fit(self, X, y=None, **kwargs):
        """
        Fit the visualizer to the specified data.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        Returns
        -------
        self : instance
            Returns the instance of the classification score visualizer

        """
        # Super fits the wrapped estimator
        super(ClassificationScoreVisualizer, self).fit(X, y)

        # Extract the classes and the class counts from the target
        self.classes_, self.class_counts_ = np.unique(y, return_counts=True)

        # Ensure the classes are aligned with the estimator
        # TODO: should we simply warn here, why would this be the case?
        if hasattr(self.estimator, 'classes_'):
            if not np.all(self.classes_ == self.estimator.classes_, axis=0):
                raise YellowbrickValueError(
                    "unique classes in y do not match estimator.classes_"
                )

        # Always return self from fit
        return self

    def score(self, X, y, **kwargs):
        """
        The score function is the hook for visual interaction. Pass in test
        data and the visualizer will create predictions on the data and
        evaluate them with respect to the test values. The evaluation will
        then be passed to draw() and the result of the estimator score will
        be returned.

        Parameters
        ----------
        X : array-like
            X (also X_test) are the dependent variables of test set to predict

        y : array-like
            y (also y_test) is the independent actual variables to score against

        Returns
        -------
        score : float
            Returns the score of the underlying model, usually accuracy for
            classification models. Refer to the specific model for more details.
        """
        # This method implements ScoreVisualizer (do not call super).
        self.score_ = self.estimator.score(X, y, **kwargs)
        return self.score_
