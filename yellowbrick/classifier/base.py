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

import warnings
import numpy as np

from yellowbrick.utils import isclassifier
from yellowbrick.base import ScoreVisualizer
from yellowbrick.style.palettes import color_palette
from yellowbrick.exceptions import NotFitted, YellowbrickWarning
from yellowbrick.exceptions import YellowbrickTypeError, ModelError

from sklearn.preprocessing import LabelEncoder


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
    estimator : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

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
        ensure you check the underlying model for more details about the metric.
    """

    def __init__(
        self,
        estimator,
        ax=None,
        fig=None,
        classes=None,
        encoder=None,
        is_fitted="auto",
        force_model=False,
        **kwargs
    ):
        # A bit of type checking
        if not force_model and not isclassifier(estimator):
            raise YellowbrickTypeError(
                "This estimator is not a classifier; "
                "try a regression or clustering score visualizer instead!"
            )

        # Initialize the super method.
        super(ClassificationScoreVisualizer, self).__init__(
            estimator, ax=ax, fig=fig, is_fitted=is_fitted, **kwargs
        )

        self.classes = classes
        self.encoder = encoder
        self.force_model = force_model

    @property
    def class_colors_(self):
        """
        Returns ``_colors`` if it exists, otherwise computes a categorical color
        per class based on the matplotlib color cycle. If the visualizer is not
        fitted, raises a NotFitted exception.

        If subclasses require users to choose colors or have specialized color
        handling, they should set ``_colors`` on init or during fit.

        Notes
        -----
        Because this is a property, this docstring is for developers only.
        """
        if not hasattr(self, "_colors"):
            if not hasattr(self, "classes_"):
                raise NotFitted("cannot determine colors before fit")

            # TODO: replace with resolve_colors
            self._colors = color_palette(None, len(self.classes_))
        return self._colors

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
        super(ClassificationScoreVisualizer, self).fit(X, y, **kwargs)

        # Extract the classes and the class counts from the target
        self.classes_, self.class_counts_ = np.unique(y, return_counts=True)

        # Ensure the classes are aligned with the estimator
        # If they are not aligned, ignore class counts and issue a warning
        if hasattr(self.estimator, "classes_"):
            if not np.array_equal(self.classes_, self.estimator.classes_):
                self.classes_ = self.estimator.classes_
                self.class_counts_ = None

        # Decode classes to human readable labels specified by the user
        self.classes_ = self._decode_labels(self.classes_)

        # Always return self from fit
        return self

    def score(self, X, y):
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
        # If the estimator has been passed in fitted but the visualizer was not fit
        # then we can retrieve the classes from the estimator, unfortunately we cannot
        # retrieve the class counts so we simply set them to None and warn the user.
        # NOTE: cannot test if hasattr(self, "classes_") because it will be proxied.
        if not hasattr(self, "class_counts_"):
            if not hasattr(self.estimator, "classes_"):
                raise NotFitted(
                    (
                        "could not determine required property classes_; "
                        "the visualizer must either be fit or instantiated with a "
                        "fitted classifier before calling score()"
                    )
                )

            self.class_counts_ = None
            self.classes_ = self._decode_labels(self.estimator.classes_)
            warnings.warn(
                "could not determine class_counts_ from previously fitted classifier",
                YellowbrickWarning,
            )

        # This method implements ScoreVisualizer (do not call super).
        self.score_ = self.estimator.score(X, y)
        return self.score_

    def _decode_labels(self, y):
        """
        An internal helper function that uses either the classes or encoder
        properties to correctly decode y as user-readable string labels.

        If both classes and encoder are set, a warning is issued and encoder is
        used instead of classes. If neither encoder nor classes is set then the
        original array is returned unmodified.
        """
        if self.classes is not None and self.encoder is not None:
            warnings.warn(
                "both classes and encoder specified, using encoder", YellowbrickWarning
            )

        if self.encoder is not None:
            # Use the label encoder or other transformer
            if hasattr(self.encoder, "inverse_transform"):
                try:
                    return self.encoder.inverse_transform(y)
                except ValueError:
                    y_labels = np.unique(y)
                    raise ModelError(
                        "could not decode {} y values to {} labels".format(
                            y_labels, self._labels()
                        )
                    )

            # Otherwise, treat as a dictionary
            try:
                return np.asarray([self.encoder[yi] for yi in y])
            except KeyError as e:
                raise ModelError(
                    (
                        "cannot decode class {} to label, "
                        "key not specified by encoder"
                    ).format(e)
                )

        if self.classes is not None:
            # Determine indices to perform class mappings on
            yp = np.asarray(y)
            if yp.dtype.kind in {"i", "u"}:
                idx = yp
            else:
                # Use label encoder to get indices by sorted class names
                idx = LabelEncoder().fit_transform(yp)

            # Use index mapping for classes
            try:
                return np.asarray(self.classes)[idx]
            except IndexError:
                y_labels = np.unique(yp)
                raise ModelError(
                    "could not decode {} y values to {} labels".format(
                        y_labels, self._labels()
                    )
                )

        # could not decode y without encoder or classes, return it as it is, unmodified
        return y

    def _labels(self):
        """
        Returns the human specified labels in either the classes list or from the
        encoder. Returns None if no human labels have been specified, but issues a
        warning if a transformer has been passed that does not specify labels.
        """
        if self.classes is not None and self.encoder is not None:
            warnings.warn(
                "both classes and encoder specified, using encoder", YellowbrickWarning
            )

        if self.encoder is not None:
            # Use label encoder or other transformer
            if hasattr(self.encoder, "transform"):
                if hasattr(self.encoder, "classes_"):
                    return self.encoder.classes_

                # This is not a label encoder
                msg = "could not determine class labels from {}".format(
                    self.encoder.__class__.__name__
                )
                warnings.warn(msg, YellowbrickWarning)
                return None

            # Otherwise, treat as dictionary and ensure sorted by key
            keys = sorted(list(self.encoder.keys()))
            return np.asarray([self.encoder[key] for key in keys])

        if self.classes is not None:
            return np.asarray(self.classes)

        return None
