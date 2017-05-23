# tests.test_classifier.test_top_features
# Testing top N features of a classifier
#
# Author:   Elaine Ayo <@ayota>
# Created:  Tue May 23 14:10:42 2017 -0400
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_top_features.py [] $

"""
Testing for the top N features visualizer
"""

import numpy as np

# from yellowbrick.classifier.top_features
from top_features import TopFeaturesVisualizer
from tests.base import VisualTestCase
from tests.dataset import DatasetMixin

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# some other model with no coef or feature importance

##########################################################################
## Data
##########################################################################

# text data (string class labels + string feature names)


# numeric data (numeric class labels + string feature names)

##########################################################################
##  Test for Top Features Visualizer
##########################################################################

## Basic tests: Text

## Basic tests: Numeric

## Test warnings/errors

## Test coef vs feature importance vs none of the above