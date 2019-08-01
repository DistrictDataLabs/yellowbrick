# tests.test_classifier.test_base
# Tests for the base classification visualizers
#
# Author:   Benjamin Bengfort
# Created:  Wed Jul 31 11:21:28 2019 -0400
#
# Copyright (C) 2019 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_base.py [] benjamin@bengfort.com $

"""
Tests for the base classification visualizers
"""

##########################################################################
## Imports
##########################################################################

import pytest

from sklearn.naive_bayes import MultinomialNB
from yellowbrick.classifier.base import *


##########################################################################
## Test Classification Score Visualizer
##########################################################################

@pytest.mark.usefixtures('binary', 'multiclass')
class TestClassificationScoreVisualizer(object):

    def test_fit(self):
        """
        Ensure that classes and class counts are computed on fit
        """

        oz = ClassificationScoreVisualizer(MultinomialNB())
        assert not hasattr(oz, 'classes_')
        assert not hasattr(oz, 'class_count_')
        assert not hasattr(oz, 'score_')
        assert oz.fit(self.binary.X, self.binary.y) is oz
        assert hasattr(oz, 'classes_')
        assert hasattr(oz, 'class_count_')
        assert not hasattr(oz, 'score_')