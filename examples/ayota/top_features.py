
"""
Implementations of visualization of top N features of text classifier

TODO:
-Docstrings
-Add tick labels
-Resolve issue with multiple plots (only produces one of five plots at the moment)
"""

##########################################################################
## Top Features Visualizer
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from operator import itemgetter

from yellowbrick.classifier import ClassificationScoreVisualizer
from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.style import color_palette

class TopFeaturesVisualizer(ClassificationScoreVisualizer):
    """
    This takes the result of a text classifier and visualizes 
    the top N positive and negative coefficients for each 
    target class in the corpus.

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
    >>> from sklearn.svm import LinearSVC
    >>> from yellowbrick.classifier import TopFeaturesVisualizer
    >>> visualizer = TopFeaturesVisualizer(LinearSVC())
    >>> visualizer.fit()
    >>> visualizer.poof()


    Notes
    -----
    This visualizer requires a text corpus. 

    """

    def __init__(self, model, ax=None, class=None, N=20, **kwargs):

        super(TopFeaturesVisualizer, self).__init__(model=model, ax=ax, **kwargs)

        self.N = N
        self.model = model

    def get_top_coefs(self, classifier, class_label, features):
        """
        Fetch top N positive and negative coefficients.

        Returns a list.
        """

        # get top n features based on class_label
        coefs = classifier.coef_[self.classes_.index(class_label)]
        top_positive_coefficients = np.argsort(coefs)[-self.N:]
        top_negative_coefficients = np.argsort(coefs)[:self.N]
        top_index = np.hstack([top_negative_coefficients, top_positive_coefficients])
        self.top_coefficients = coefs[top_index]
        self.top_features = [features[i] for i in top_index]


    def fit(self, X_train, y, **kwargs):
        """
        Pull in all data transform methods and apply to data here.

        Then, make plots for desired classes.
        """

        super(TopFeaturesVisualizer, self).fit(X_train, y, **kwargs)
        self.classes_ = self.estimator.classes_

        return self

    def score(self, features, class_label, **kwargs):
        """
        Pull in all data transform methods and apply to data here.

        Then, make plots for desired classes.
        """
        
        self.class_label = class_label
        # can get features from estimator?
        self.get_top_coefs(self.estimator, class_label, features)
        self.draw()

    def draw(self, **kwargs):
        """
        Draw basic plot here, with all finishing for each individual subplot called here.
        """
        # Create the axis if it doesn't exist
        if self.ax is None:
            self.ax = plt.gca()

        palette = color_palette(n_colors=2)
        colors = [palette[0] if c < 0 else palette[1] for c in self.top_coefficients]
        self.ax.bar(np.arange(2 * self.N), self.top_coefficients, color=colors)
        self.ax.set_xticks(np.arange(1, 1 + 2 * self.N))


    def finalize(self, **kwargs):
        """
        Add final touches: tick labels and title.
        """

        # Add a title
        self.set_title('Top {N} features for {class_label}'.format(N=self.N, class_label=self.class_label))
        
        # Add tick labels
        self.ax.set_xticklabels(self.top_features, rotation=90, ha='right')
