# tests.test_text.test_freqdist
# Tests for the frequency distribution visualization
#
# Author:   Rebecca Bilbro
# Created:  2017-03-22 15:27
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_freqdist.py [bd9cbb9] rebecca.bilbro@bytecubed.com $

"""
Tests for the frequency distribution text visualization
"""

##########################################################################
## Imports
##########################################################################

import pytest

from yellowbrick.datasets import load_hobbies
from yellowbrick.text.freqdist import *
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase

from sklearn.feature_extraction.text import CountVectorizer

##########################################################################
## Data
##########################################################################

corpus = load_hobbies()

##########################################################################
## FreqDist Tests
##########################################################################


class TestFreqDist(VisualTestCase):
    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_integrated_freqdist(self):
        """
        Assert no errors occur during freqdist integration
        """
        vectorizer = CountVectorizer()

        docs = vectorizer.fit_transform(corpus.data)
        features = vectorizer.get_feature_names()

        visualizer = FreqDistVisualizer(features)
        visualizer.fit(docs)

        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.5)  # w/o tol fails with RMS 0.121

    def test_freqdist_quickmethod(self):
        """
        Assert no errors occur during freqdist quickmethod
        """
        vectorizer = CountVectorizer()

        docs = vectorizer.fit_transform(corpus.data)
        features = vectorizer.get_feature_names()

        viz = freqdist(features, docs, show=False)

        assert isinstance(viz, FreqDistVisualizer)

        # yellowbrick.exceptions.ImageComparisonFailure: images not close (RMS 1.401)
        self.assert_images_similar(viz, tol=1.5)
