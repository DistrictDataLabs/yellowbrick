# classifier.top_features
# Visualize top N features of a classifier
#
# Author:   Elaine Ayo <@ayota>
# Created:  Tue May 23 14:10:42 2017 -0400
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: classifier.top_features.base.py [] $


"""
Visualize the top N features of classifier.

TODO:
-Docstrings
-Add tick labels
-Make quick method for producing plots for all classes using multiclass visualizer
-Add support for estimators with _feature_importance attributes
-Implement one-sided version that takes absolute value of coefs/features
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import warnings
import matplotlib.pyplot as plt

from yellowbrick.classifier import ClassificationScoreVisualizer
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.style import color_palette

##########################################################################
## Top Features Visualizer
##########################################################################

class TopFeaturesVisualizer(ClassificationScoreVisualizer):
    """
    This takes the result of a classifier and visualizes
    the top N positive and negative coefficients for a target class.
    
    Scikit-learn estimators store this information in two attributes:
    coef_ and feature_importances_.
    
    Any classifier given to this visualizer must have coef_, 
    otherwise we'll throw a "Estimator Not Implemented" warning.
    
    TODO:
    * Implement visualizer for feature_importances_
    * Add funcitonality for "one sided" visualizer that takes absolute values
    and ranks those.

    Parameters
    ----------
    ax : matplotlib axes, default: None
        The axes to plot the figure on. If None is passed in, the
        current axes will be used (or generated if required).


    N : int
        The number of features to visualize among the positive and
        negative coefficients.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------
    # Text Example
    >>> from sklearn.svm import LinearSVC
    >>> from yellowbrick.classifier import TopFeaturesVisualizer
    >>> visualizer = TopFeaturesVisualizer(LinearSVC())
    >>> visualizer.fit()
    >>> visualizer.poof()
    
    # Numeric Example
    >>> # numeric example here!!!!


    Notes
    -----
    Text corpora need to be formatted as a vectorizer and 
    the top features extracted before using the visualizer.
    
    data = corpus.data
    target = corpus.target
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(data)
    X_train = vectorizer.transform(data)
    features = vectorizer.get_feature_names()

    """

    def __init__(self, model, ax=None, N=20, features=None, **kwargs):
        # TODO: Make sure model is a classifier; otherwise throw a warning.

        super(TopFeaturesVisualizer, self).__init__(model=model, ax=ax, **kwargs)

        self.N = N
        self.features = features


    def fit(self, X_train, y, **kwargs):
        """
        Pull in all data transform methods and apply to data here.
        """

        self.estimator.fit(X_train, y, **kwargs)
        self.classes_ = self.estimator.classes_

        return self

    def score(self, class_label=None, **kwargs):
        """
        Extract top N features for desired class label.
        
        Call draw().
        """

        if self.ax is None:
            self.ax = plt.gca()

        # raise error is no class label provided
        if class_label is None:
            raise YellowbrickValueError(
                "Please provide a class label."
                )

        self.class_label = class_label
        self._get_top_features(class_label=self.class_label, **kwargs)
        self.draw()


    def draw(self, **kwargs):
        """
        Draw basic plot.
        """

        palette = color_palette(n_colors=2)
        colors = [palette[0] if c < 0 else palette[1] for c in self.top_coefficients]
        self.ax.bar(np.arange(2 * self.N), self.top_coefficients, color=colors)
        self.ax.set_xticks(np.arange(1, 1 + 2 * self.N))


    def finalize(self, **kwargs):
        """
        Add final touches: tick labels and title.
        """

        self.set_title('Top {N} features for {class_label}'.format(N=self.N, class_label=self.class_label))
        self.ax.set_xticklabels(self.top_features, rotation=90, ha='right')

    def _get_top_param(self):
        """
        Searches for the parameter on the estimator that contains the array of
        coefficients or features with the most "importance."

        Right now, this only checks for the coef_ attribute; feature_importance TK. 
        If the estimator has neither of these, raise YellowbrickValueError.
        """
        try:
            return getattr(self.estimator, 'coef_')
        except AttributeError:
            raise YellowbrickValueError(
                "Could not find coef_ param on {} estimator. "
                "Please select estimator with coef_ attribute.".format(
                    self.estimator.__class__.__name__
                )
            )

    def _get_top_features(self, class_label=None, features=None):
        """
        Fetch top N positive and negative coefficients for given class label.

        Also fetches corresponding names of features if they exist; otherwise
        assign index numbers and throw a warning.
        
        Throws warning if no class label defined.
        """
        # check for user-provided feature in visualizer init
        if self.features:
            features = self.features

        # fetch top feature values, get values for desired class label
        values = self._get_top_param()
        coefs = values[np.where(self.classes_ == class_label)[0][0]]
        top_positive_coefficients = np.argsort(coefs)[-self.N:]
        top_negative_coefficients = np.argsort(coefs)[:self.N]
        top_index = np.hstack([top_negative_coefficients, top_positive_coefficients])
        self.top_coefficients = coefs[top_index]

        # get pretty names if they exist
        try:
            self.top_features = [features[i] for i in top_index]
        except TypeError:
            self.top_features = top_index
            warnings.warn(
                "No feature names specified. Index values will be used."
            )

##########################################################################
## Top Features Visualizer Quick Method
##########################################################################

# It would be nice if there were a function here that used the top features
# visualizer to grab the top features for each class and output them
# using multiplot visualizer.

