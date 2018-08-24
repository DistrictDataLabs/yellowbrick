# feature_correlation
# Generates images for the feature correlation documentation.
#
# Author:  Zijie (ZJ) Poh <poh.zijie@gmail.com>
# Created: Tue Jul 31 20:21:32 2018 -0700
#
#
"""
Generates images for the feature correlation documentation.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import pandas as pd
from sklearn import datasets

from yellowbrick.target import FeatureCorrelation


##########################################################################
## Plotting Functions
##########################################################################

def feature_correlation_pearson(
        path="images/feature_correlation_pearson.png"):
    data = datasets.load_diabetes()
    X, y = data['data'], data['target']
    feature_names = np.array(data['feature_names'])

    visualizer = FeatureCorrelation(labels=feature_names)
    visualizer.fit(X, y)
    visualizer.poof(outpath=path, clear_figure=True)


def feature_correlation_mutual_info_classification(
        path="images/feature_correlation_mutual_info_classification.png"):
    data = datasets.load_wine()
    X, y = data['data'], data['target']
    feature_names = np.array(data['feature_names'])
    X_pd = pd.DataFrame(X, columns=feature_names)

    feature_to_plot = ['alcohol', 'ash', 'hue', 'proline', 'total_phenols']

    visualizer = FeatureCorrelation(method='mutual_info-classification',
                                    feature_names=feature_to_plot)
    visualizer.fit(X_pd, y, random_state=0)
    visualizer.poof(outpath=path, clear_figure=True)


def feature_correlation_mutual_info_regression(
        path="images/feature_correlation_mutual_info_regression.png"):
    data = datasets.load_diabetes()
    X, y = data['data'], data['target']
    feature_names = np.array(data['feature_names'])

    discrete_features = [False for _ in range(len(feature_names))]
    discrete_features[1] = True

    visualizer = FeatureCorrelation(method='mutual_info-regression',
                                    labels=feature_names, sort=True)
    visualizer.fit(X, y, discrete_features=discrete_features, random_state=0)
    visualizer.poof(outpath=path, clear_figure=True)


if __name__ == '__main__':
    feature_correlation_pearson()
    feature_correlation_mutual_info_classification()
    feature_correlation_mutual_info_regression()
