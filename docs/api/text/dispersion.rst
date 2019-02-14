.. -*- mode: rst -*-

Dispersion Plot
===============

A word's importance can be weighed by its dispersion in a corpus.  Lexical dispersion is a measure of a word's homogeneity across the parts of a corpus.  This plot notes the occurrences of a word and how many words from the beginning of the corpus it appears.

.. plot::
    :context: close-figs

    from yellowbrick.text import DispersionPlot
    from yellowbrick.datasets import load_hobbies

    # Load the text data
    corpus = load_hobbies()

    # Create a list of words from the corpus text
    text = [doc.split() for doc in corpus.data]

    # Choose words whose occurence in the text will be plotted
    target_words = ['Game', 'player', 'score', 'oil', 'Man']

    # Create the visualizer and draw the plot
    visualizer = DispersionPlot(target_words)
    visualizer.fit(text)
    visualizer.poof()


API Reference
-------------

.. automodule:: yellowbrick.text.dispersion
    :members: DispersionPlot
    :undoc-members:
    :show-inheritance:
