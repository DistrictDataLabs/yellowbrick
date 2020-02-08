.. -*- mode: rst -*-

Token Frequency Distribution
============================

A method for visualizing the frequency of tokens within and across corpora is frequency distribution. A frequency distribution tells us the frequency of each vocabulary item in the text. In general, it could count any kind of observable event. It is a distribution because it tells us how the total number of word tokens in the text are distributed across the vocabulary items.

=================   ==============================
Visualizer           :class:`~yellowbrick.text.freqdist.FrequencyVisualizer`
Quick Method         :func:`~yellowbrick.text.freqdist.freqdist`
Models               Text
Workflow             Text Analysis
=================   ==============================

.. NOTE:: The ``FreqDistVisualizer`` does not perform any normalization or vectorization, and it expects text that has already been count vectorized.

We first instantiate a ``FreqDistVisualizer`` object, and then call ``fit()`` on that object with the count vectorized documents and the features (i.e. the words from the corpus), which computes the frequency distribution. The visualizer then plots a bar chart of the top 50 most frequent terms in the corpus, with the terms listed along the x-axis and frequency counts depicted at y-axis values. As with other Yellowbrick visualizers, when the user invokes ``show()``, the finalized visualization is shown.  Note that in this plot and in the subsequent one, we can orient our plot vertically by passing in ``orient='v'`` on instantiation (the plot will orient horizontally by default):

.. plot::
    :context: close-figs
    :alt: Frequency Distribution Plot

    from sklearn.feature_extraction.text import CountVectorizer

    from yellowbrick.text import FreqDistVisualizer
    from yellowbrick.datasets import load_hobbies

    # Load the text data
    corpus = load_hobbies()

    vectorizer = CountVectorizer()
    docs       = vectorizer.fit_transform(corpus.data)
    features   = vectorizer.get_feature_names()

    visualizer = FreqDistVisualizer(features=features, orient='v')
    visualizer.fit(docs)
    visualizer.show()


It is interesting to compare the results of the ``FreqDistVisualizer`` before and after stopwords have been removed from the corpus:


.. plot::
    :context: close-figs
    :include-source: False
    :alt: Frequency Distribution Plot without Stopwords

    from sklearn.feature_extraction.text import CountVectorizer

    from yellowbrick.text import FreqDistVisualizer
    from yellowbrick.datasets import load_hobbies

    # Load the text data
    corpus = load_hobbies()

    vectorizer = CountVectorizer(stop_words='english')
    docs       = vectorizer.fit_transform(corpus.data)
    features   = vectorizer.get_feature_names()

    visualizer = FreqDistVisualizer(features=features, orient='v')
    visualizer.fit(docs)
    visualizer.show()

It is also interesting to explore the differences in tokens across a corpus. The hobbies corpus that comes with Yellowbrick has already been categorized (try ``corpus.target``), so let's visually compare the differences in the frequency distributions for two of the categories: *"cooking"* and *"gaming"*.

Here is the plot for the cooking corpus (oriented horizontally this time):

.. plot::
    :context: close-figs
    :include-source: False
    :alt: Frequency Distribution Plot for Cooking Corpus

    from collections import defaultdict

    from sklearn.feature_extraction.text import CountVectorizer

    from yellowbrick.text import FreqDistVisualizer
    from yellowbrick.datasets import load_hobbies

    # Load the text data
    corpus = load_hobbies()

    # Create a dict to map target labels to documents of that category
    hobbies = defaultdict(list)
    for text, label in zip(corpus.data, corpus.target):
        hobbies[label].append(text)

    vectorizer = CountVectorizer(stop_words='english')
    docs       = vectorizer.fit_transform(text for text in hobbies['cooking'])
    features   = vectorizer.get_feature_names()

    visualizer = FreqDistVisualizer(
        features=features, size=(1080, 720)
    )
    visualizer.fit(docs)
    visualizer.show()

And for the gaming corpus (again oriented horizontally):

.. plot::
    :context: close-figs
    :include-source: False
    :alt: Frequency Distribution Plot for Gaming Corpus

    from collections import defaultdict

    from sklearn.feature_extraction.text import CountVectorizer

    from yellowbrick.text import FreqDistVisualizer
    from yellowbrick.datasets import load_hobbies

    # Load the text data
    corpus = load_hobbies()

    # Create a dict to map target labels to documents of that category
    hobbies = defaultdict(list)
    for text, label in zip(corpus.data, corpus.target):
        hobbies[label].append(text)

    vectorizer = CountVectorizer(stop_words='english')
    docs       = vectorizer.fit_transform(text for text in hobbies['gaming'])
    features   = vectorizer.get_feature_names()

    visualizer = FreqDistVisualizer(
        features=features, size=(1080, 720)
    )
    visualizer.fit(docs)
    visualizer.show()

Quick Method
------------

Similar functionality as above can be achieved in one line using the associated quick method, ``freqdist``. This method will instantiate with features(words) and fit a ``FreqDistVisualizer`` visualizer on the documents).

.. plot::
    :context: close-figs
    :alt: Frequency Distribution quick method

    from collections import defaultdict

    from sklearn.feature_extraction.text import CountVectorizer

    from yellowbrick.text import freqdist
    from yellowbrick.datasets import load_hobbies

    # Load the text data
    corpus = load_hobbies()

    # Create a dict to map target labels to documents of that category
    hobbies = defaultdict(list)
    for text, label in zip(corpus.data, corpus.target):
        hobbies[label].append(text)

    vectorizer = CountVectorizer(stop_words='english')
    docs       = vectorizer.fit_transform(text for text in hobbies['cinema'])
    features   = vectorizer.get_feature_names()

    freqdist(features, docs, orient='v')


API Reference
-------------

.. automodule:: yellowbrick.text.freqdist
    :members: FrequencyVisualizer, freqdist
    :undoc-members:
    :show-inheritance:
