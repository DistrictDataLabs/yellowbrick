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


import unittest
import numpy as np

from yellowbrick.text.tsne import *
from tests.dataset import DatasetMixin
from yellowbrick.exceptions import YellowbrickValueError
from sklearn.feature_extraction.text import TfidfVectorizer


##########################################################################
## TSNE Tests
##########################################################################

class TSNETests(unittest.TestCase, DatasetMixin):

    def test_bad_decomposition(self):
        """
        Ensure an error is raised when a bad decompose argument is specified
        """
        with self.assertRaises(YellowbrickValueError):
            tsne = TSNEVisualizer(decompose='bob')

    def test_make_pipeline(self):
        """
        Verify the pipeline creation step for TSNE
        """

        tsne = TSNEVisualizer() # Should not cause an exception.
        self.assertIsNotNone(tsne.transformer_)

        svdp = tsne.make_transformer('svd', 90)
        self.assertEqual(len(svdp.steps), 2)

        pcap = tsne.make_transformer('pca')
        self.assertEqual(len(pcap.steps), 2)

        none = tsne.make_transformer(None)
        self.assertEqual(len(none.steps), 1)

    def test_integrated_tsne(self):
        """
        Assert no errors occur during tsne integration
        """
        corpus = self.load_data('hobbies')
        tfidf  = TfidfVectorizer()

        docs   = tfidf.fit_transform(corpus.data)
        labels = corpus.target 

        tsne = TSNEVisualizer()
        tsne.fit_transform(docs, labels)
