.. -*- mode: rst -*-

Word Correlation Plot
=====================

Word correlation illustrates the extent to which words or phrases co-appear across the documents in a corpus. This can be useful for understanding the relationships between known text features in a corpus with many documents. ``WordCorrelationPlot`` allows for the visualization of the document occurrence correlations between select words in a corpus. For a number of features n, the plot renders an n x n heatmap containing correlation values.

The correlation values are computed using the `phi coefficient <https://en.wikipedia.org/wiki/Phi_coefficient>`_ metric, which is a measure of the association between two binary variables. A value close to 1 or -1 indicates that the occurrences of the two features are highly positively or negatively correlated, while a value close to 0 indicates no relationship between the two features.

=================   ==============================
Visualizer           :class:`~yellowbrick.text.correlation.WordCorrelationPlot`
Quick Method         :func:`~yellowbrick.text.correlation.word_correlation()`
Models               Text Modeling
Workflow             Feature Engineering
=================   ==============================

.. plot::
    :context: close-figs
    :alt: Word Correlation Plot

    from yellowbrick.datasets import load_hobbies
    from yellowbrick.text.correlation import WordCorrelationPlot

    # Load the text corpus
    corpus = load_hobbies()

    # Create the list of words to plot
    words = ["Tatsumi Kimishima", "Nintendo", "game", "play", "man", "woman"]

    # Instantiate the visualizer and draw the plot
    viz = WordCorrelationPlot(words)
    viz.fit(corpus.data)
    viz.show()


Quick Method
------------

The same functionality above can be achieved with the associated quick method `word_correlation`. This method will build the Word Correlation Plot object with the associated arguments, fit it, then (optionally) immediately show the visualization.

.. plot::
    :context: close-figs
    :alt: Word Correlation Plot

    from yellowbrick.datasets import load_hobbies
    from yellowbrick.text.correlation import word_correlation

    # Load the text corpus
    corpus = load_hobbies()

    # Create the list of words to plot
    words = ["Game", "player", "score", "oil"]

    # Draw the plot
    word_correlation(words, corpus.data)

API Reference
-------------

.. automodule:: yellowbrick.text.correlation
    :members: WordCorrelationPlot, word_correlation
    :undoc-members:
    :show-inheritance:
