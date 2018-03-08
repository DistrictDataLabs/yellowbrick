.. -*- mode: rst -*-

Text Modeling Visualizers
=========================

Yellowbrick provides the `yellowbrick.text` module for text-specific visualizers. The `TextVisualizer` class specifically deals with datasets that are corpora and not simple numeric arrays or DataFrames, providing utilities for analyzing word distribution, showing document similarity, or simply wrapping some of the other standard visualizers with text-specific display properties.

We currently have two text-specific visualizations implemented:

-  :doc:`freqdist`: plot the frequency of tokens in a corpus
-  :doc:`tsne`: plot similar documents closer together to discover clusters

Note that the examples in this section require a corpus of text data, see :doc:`loading a text corpus <corpus>` for more information.

.. code:: python

    from yellowbrick.text import FreqDistVisualizer
    from yellowbrick.text import TSNEVisualizer

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer

.. toctree::
   :maxdepth: 2

   corpus
   freqdist
   tsne
