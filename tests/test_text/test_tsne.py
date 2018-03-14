# tests.test_text.test_tsne
# Tests for the TSNE visual corpus embedding mechanism.
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Mon Feb 20 07:23:53 2017 -0500
#
# Copyright (C) 2016 Bengfort.com
# For license information, see LICENSE.txt
#
# ID: test_tsne.py [6aa9198] benjamin@bengfort.com $

"""
Tests for the TSNE visual corpus embedding mechanism.
"""

##########################################################################
## Imports
##########################################################################

import six 
import pytest

from yellowbrick.text.tsne import *
from tests.base import VisualTestCase
from tests.dataset import DatasetMixin
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer


##########################################################################
## TSNE Tests
##########################################################################

class TestTSNE(VisualTestCase, DatasetMixin):
    """
    TSNEVisualizer tests
    """

    def test_bad_decomposition(self):
        """
        Ensure an error is raised when a bad decompose argument is specified
        """
        with pytest.raises(YellowbrickValueError):
            TSNEVisualizer(decompose='bob')

    def test_make_pipeline(self):
        """
        Verify the pipeline creation step for TSNE
        """

        tsne = TSNEVisualizer() # Should not cause an exception.
        assert tsne.transformer_ is not None

        svdp = tsne.make_transformer('svd', 90)
        assert len(svdp.steps) == 2

        pcap = tsne.make_transformer('pca')
        assert len(pcap.steps) == 2

        none = tsne.make_transformer(None)
        assert len(none.steps) == 1

    def test_integrated_tsne(self):
        """
        Check tSNE integrated visualization on the hobbies corpus
        """
        corpus = self.load_data('hobbies')
        tfidf  = TfidfVectorizer()

        docs   = tfidf.fit_transform(corpus.data)
        labels = corpus.target

        tsne = TSNEVisualizer(random_state=8392, colormap='Set1')
        tsne.fit_transform(docs, labels)
        
        tol = 40 if six.PY3 else 55
        self.assert_images_similar(tsne, tol=tol)

    def test_make_classification_tsne(self):
        """
        Test tSNE integrated visualization on a sklearn classifier dataset
        """

        ## produce random data
        X, y = make_classification(n_samples=200, n_features=100,
                               n_informative=20, n_redundant=10,
                               n_classes=3, random_state=42)

        ## visualize data with t-SNE
        tsne = TSNEVisualizer(random_state=87)
        tsne.fit(X, y)
    
        tol = 0.1 if six.PY3 else 40
        self.assert_images_similar(tsne, tol=tol)

    def test_make_classification_tsne_class_labels(self):
        """
        Test tSNE integrated visualization with class labels specified
        """

        ## produce random data
        X, y = make_classification(n_samples=200, n_features=100,
                               n_informative=20, n_redundant=10,
                               n_classes=3, random_state=42)

        ## visualize data with t-SNE
        tsne = TSNEVisualizer(random_state=87, labels=['a', 'b', 'c'])
        tsne.fit(X, y)
    
        tol = 0.1 if six.PY3 else 40
        self.assert_images_similar(tsne, tol=tol)

    def test_tsne_mismtached_labels(self):
        """
        Assert exception is raised when number of labels doesn't match
        """
        ## produce random data
        X, y = make_classification(n_samples=200, n_features=100,
                               n_informative=20, n_redundant=10,
                               n_classes=3, random_state=42)

        ## fewer labels than classes
        tsne = TSNEVisualizer(random_state=87, labels=['a', 'b'])
        with pytest.raises(YellowbrickValueError):
            tsne.fit(X,y)

        ## more labels than classes
        tsne = TSNEVisualizer(random_state=87, labels=['a', 'b', 'c', 'd'])
        with pytest.raises(YellowbrickValueError):
            tsne.fit(X,y)


    def test_no_target_tsne(self):
        """
        Test tSNE when no target or classes are specified
        """
        ## produce random data
        X, y = make_classification(n_samples=200, n_features=100,
                               n_informative=20, n_redundant=10,
                               n_classes=3, random_state=6897)

        ## visualize data with t-SNE
        tsne = TSNEVisualizer(random_state=64)
        tsne.fit(X)

        self.assert_images_similar(tsne, tol=0.1)
