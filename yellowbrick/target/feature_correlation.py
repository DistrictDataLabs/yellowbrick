# yellowbrick.classifier.feature_correlation
# Class balance visualizer for showing per-class support.
#
# Author    Zijie (ZJ) Poh <poh.zijie@gmail.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Wed Jul 29 15:30:40 2018 -0700
#
# ID: feature_correlation.py [5388065] neal@nhumphrey.com $

"""
Feature Correlation to Dependent Variable Visualizer.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from .base import TargetVisualizer

from sklearn.feature_selection import (mutual_info_classif,
                                       mutual_info_regression)
from scipy.stats import pearsonr


##########################################################################
## Class Feature Correlation
##########################################################################

class FeatureCorrelation(TargetVisualizer):

    method_func = {
        'pearson': lambda X, y: [pearsonr(x, y)[0] for x in X.T],
        'mutual_info': lambda X, y: mutual_info_regression(X, y),
        'mutual_info_classif': lambda X, y: mutual_info_classif(X, y)
    }
    method_label = {
        'pearson': 'Pearson Correlation',
        'mutual_info': 'Mutual Information',
        'mutual_info_classif': 'Mutual Information'
    }

    def __init__(self, ax=None, method='pearson', classification=False,
                 **kwargs):
        super(FeatureCorrelation, self).__init__(ax=None, **kwargs)

        if classification and method == 'mutual_info':
            method = 'mutual_info_classif'

        # Parameters
        self.set_params(
            method=method
        )

    def fit(self, X, y, features_):

        # Calculate Features correlation with target variable
        self.coef_ = np.array(self.method_func[self.method](X, y))

        # Sort features by correlation
        sort_idx = np.argsort(self.coef_)
        self.coef_ = self.coef_[sort_idx]
        self.features_ = features_[sort_idx]

        self.draw()
        return self

    def draw(self):

        pos = np.arange(self.coef_.shape[0]) + 0.5

        self.ax.barh(pos, self.coef_)

        # Set the labels for the bars
        self.ax.set_yticks(pos)
        self.ax.set_yticklabels(self.features_)

        return self.ax

    def finalize(self, **kwargs):

        self.set_title('Features correlation with dependent variable')

        self.ax.set_xlabel(self.method_label[self.method])

        self.ax.grid(False, axis='y')
