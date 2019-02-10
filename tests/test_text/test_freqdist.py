# tests.test_text.test_freqdist
# Tests for the frequency distribution visualization
#
# Author:   Rebecca Bilbro
# Created:  2017-03-22 15:27
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_freqdist.py [bd9cbb9] $

"""
Tests for the frequency distribution text visualization
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest

from yellowbrick.datasets import load_hobbies
from yellowbrick.text.freqdist import *
from tests.base import VisualTestCase

from sklearn.feature_extraction.text import CountVectorizer

##########################################################################
## Data
##########################################################################

corpus = load_hobbies()

##########################################################################
## FreqDist Tests
##########################################################################

class FreqDistTests(VisualTestCase):

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_integrated_freqdist(self):
        """
        Assert no errors occur during freqdist integration
        """
        vectorizer = CountVectorizer()

        docs       = vectorizer.fit_transform(corpus.data)
        features   = vectorizer.get_feature_names()

        visualizer = FreqDistVisualizer(features)
        visualizer.fit(docs)

        visualizer.poof()
        self.assert_images_similar(visualizer, tol=1)
