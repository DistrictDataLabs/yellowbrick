.. -*- mode: rst -*-

Dispersion Plot
===============

A word's importance can be weighed by its dispersion in a corpus. Lexical dispersion is a measure of a word's homogeneity across the parts of a corpus.

Lexical dispersion illustrates the homogeneity of a word (or set of words) across
the documents of a corpus. ``DispersionPlot`` allows for visualization of the lexical dispersion of words in a corpus. This plot illustrates with vertical lines the occurrences of one or more search terms throughout the corpus, noting how many words relative to the beginning of the corpus it appears.

=================   ==============================
Visualizer           :class:`~yellowbrick.text.dispersion.DispersionPlot`
Quick Method         :func:`~yellowbrick.text.dispersion.dispersion()`
Models               Text Modeling
Workflow             Feature Engineering
=================   ==============================

.. plot::
    :context: close-figs
    :alt: Dispersion Plot

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
    visualizer.show()

If the target vector of the corpus documents is provided, the points will be colored with respect to their document category, which allows for additional analysis of relationships in search term homogeneity within and across document categories.

.. plot::
    :context: close-figs
    :alt: Dispersion Plot with Classes

    from yellowbrick.text import DispersionPlot
    from yellowbrick.datasets import load_hobbies

    corpus = load_hobbies()
    text = [doc.split() for doc in corpus.data]
    y = corpus.target

    target_words = ['points', 'money', 'score', 'win', 'reduce']

    visualizer = DispersionPlot(
        target_words,
        colormap="Accent",
        title="Lexical Dispersion Plot, Broken Down by Class"
    )
    visualizer.fit(text, y)
    visualizer.show()


Quick Method
------------

The same functionality above can be achieved with the associated quick method `dispersion`. This method will build the Dispersion Plot object with the associated arguments, fit it, then (optionally) immediately show the visualization.

.. plot::
    :context: close-figs
    :alt: Quick Method Dispersion Plot

    from yellowbrick.text import DispersionPlot, dispersion
    from yellowbrick.datasets import load_hobbies

    # Load the text data
    corpus = load_hobbies()

    # Create a list of words from the corpus text
    text = [doc.split() for doc in corpus.data]

    # Choose words whose occurence in the text will be plotted
    target_words = ['features', 'mobile', 'cooperative', 'competitive', 'combat', 'online']

    # Create the visualizer and draw the plot
    dispersion(target_words, text, colors=['olive'])


API Reference
-------------

.. automodule:: yellowbrick.text.dispersion
    :members: DispersionPlot, dispersion
    :undoc-members:
    :show-inheritance:
