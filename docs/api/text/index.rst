.. -*- mode: rst -*-

Text Modeling Visualizers
=========================

Yellowbrick provides the ``yellowbrick.text`` module for text-specific visualizers. The ``TextVisualizer`` class specifically deals with datasets that are corpora and not simple numeric arrays or DataFrames, providing utilities for analyzing word dispersion and distribution, showing document similarity, or simply wrapping some of the other standard visualizers with text-specific display properties.

We currently have five text-specific visualizations implemented:

-  :doc:`freqdist`: plot the frequency of tokens in a corpus
-  :doc:`tsne`: plot similar documents closer together to discover clusters
-  :doc:`umap_vis`: plot similar documents closer together to discover clusters
-  :doc:`dispersion`: plot the dispersion of target words throughout a corpus
-  :doc:`postag`: plot the counts of different parts-of-speech throughout a tagged corpus

Note that the examples in this section require a corpus of text data, see :doc:`the hobbies corpus <../datasets/hobbies>` for a sample dataset.

.. code:: python

    from yellowbrick.text import FreqDistVisualizer
    from yellowbrick.text import TSNEVisualizer
    from yellowbrick.text import UMAPVisualizer
    from yellowbrick.text import DispersionPlot
    from yellowbrick.text import PosTagVisualizer

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer

.. toctree::
   :maxdepth: 2

   freqdist
   tsne
   umap_vis
   dispersion
   postag