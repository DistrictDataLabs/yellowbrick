.. -*- mode: rst -*-

t-SNE Corpus Visualization
==========================

=================   =================
Visualizer           :class:`~yellowbrick.text.tsne.TSNEVisualizer`
Quick Method         :func:`~yellowbrick.text.tsne.tsne`
Models               Decomposition
Workflow             Feature Engineering/Selection
=================   =================

One very popular method for visualizing document similarity is to use t-distributed stochastic neighbor embedding, t-SNE. Scikit-learn implements this decomposition method as the ``sklearn.manifold.TSNE`` transformer. By decomposing high-dimensional document vectors into 2 dimensions using probability distributions from both the original dimensionality and the decomposed dimensionality, t-SNE is able to effectively cluster similar documents. By decomposing to 2 or 3 dimensions, the documents can be visualized with a scatter plot.

Unfortunately, ``TSNE`` is very expensive, so typically a simpler decomposition method such as SVD or PCA is applied ahead of time. The ``TSNEVisualizer`` creates an inner transformer pipeline that applies such a decomposition first (SVD with 50 components by default), then performs the t-SNE embedding. The visualizer then plots the scatter plot, coloring by cluster or by class, or neither if a structural analysis is required.

After importing the required tools, we can use the :doc:`hobbies corpus <../datasets/hobbies>` and vectorize the text using TF-IDF. Once the corpus is vectorized we can visualize it, showing the distribution of classes.

.. plot::
    :context: close-figs
    :alt: TSNE Plot

    from sklearn.feature_extraction.text import TfidfVectorizer

    from yellowbrick.text import TSNEVisualizer
    from yellowbrick.datasets import load_hobbies

    # Load the data and create document vectors
    corpus = load_hobbies()
    tfidf = TfidfVectorizer()

    X = tfidf.fit_transform(corpus.data)
    y = corpus.target

    # Create the visualizer and draw the vectors
    tsne = TSNEVisualizer()
    tsne.fit(X, y)
    tsne.show()

Note that you can pass the class labels or document categories directly to the ``TSNEVisualizer`` as follows:

.. code:: python

    labels = corpus.labels
    tsne = TSNEVisualizer(labels=labels)
    tsne.fit(X, y)
    tsne.show()

If we omit the target during fit, we can visualize the whole dataset to see if any meaningful patterns are observed.

.. plot::
    :context: close-figs
    :include-source: False
    :alt: TSNE Plot without Class Coloring

    from sklearn.feature_extraction.text import TfidfVectorizer

    from yellowbrick.text import TSNEVisualizer
    from yellowbrick.datasets import load_hobbies

    # Load the data and create document vectors
    corpus = load_hobbies()
    tfidf = TfidfVectorizer()

    X = tfidf.fit_transform(corpus.data)
    tsne = TSNEVisualizer(labels=["documents"])
    tsne.fit(X)
    tsne.show()

This means we don't have to use class labels at all. Instead we can use cluster membership from K-Means to label each document. This will allow us to look for clusters of related text by their contents:

.. plot::
    :context: close-figs
    :include-source: False
    :alt: TSNE Plot without Clustering

    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    from yellowbrick.text import TSNEVisualizer
    from yellowbrick.datasets import load_hobbies

    # Load the data and create document vectors
    corpus = load_hobbies()
    tfidf = TfidfVectorizer()

    X = tfidf.fit_transform(corpus.data)

    clusters = KMeans(n_clusters=5)
    clusters.fit(X)

    tsne = TSNEVisualizer()
    tsne.fit(X, ["c{}".format(c) for c in clusters.labels_])
    tsne.show()

Quick Method
------------

The same functionality above can be achieved with the associated quick method ``tsne``. This method will build the ``TSNEVisualizer`` object with the associated arguments, fit it, then (optionally) immediately show it

.. plot::
    :context: close-figs
    :alt: tsne quick method on the hobbies dataset

    from yellowbrick.text.tsne import tsne
    from sklearn.feature_extraction.text import TfidfVectorizer
    from yellowbrick.datasets import load_hobbies

    # Load the data and create document vectors
    corpus = load_hobbies()
    tfidf = TfidfVectorizer()

    X = tfidf.fit_transform(corpus.data)
    y = corpus.target

    tsne(X, y)


API Reference
-------------

.. automodule:: yellowbrick.text.tsne
    :members: TSNEVisualizer, tsne
    :undoc-members:
    :show-inheritance:
