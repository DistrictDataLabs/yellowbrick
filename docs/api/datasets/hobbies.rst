.. -*- mode: rst -*-

Hobbies
=======

The Baleen hobbies corpus contains 448 files in 5 categories.

=================   ==========================================================
Samples total                                                              448
Dimensionality                                                           23738
Features                                                      strings (tokens)
Targets                str: {"books", "cinema", "cooking", "gaming", "sports"}
Task(s)                                             classification, clustering
=================   ==========================================================

Description
-----------

The hobbies corpus is a text corpus wrangled from the `Baleen RSS Corpus <https://github.com/DistrictDataLabs/baleen>`_ in order to enable students and readers to practice different techniques in Natural Language Processing. For more information see `Applied Text Analysis with Python: Enabling Language-Aware Data Products with Machine Learning <https://www.amazon.com/Applied-Text-Analysis-Python-Language-Aware/dp/1491963042>`_ and the associated `code repository <https://github.com/foxbook/atap>`_. It is structured as:

Documents and File Size
~~~~~~~~~~~~~~~~~~~~~~~

- books: 72 docs (4.1MiB)
- cinema: 100 docs (9.2MiB)
- cooking: 30 docs (3.0MiB)
- gaming: 128 docs (8.8MiB)
- sports: 118 docs (15.9MiB)

Document Structure
~~~~~~~~~~~~~~~~~~

Overall:

- 7,420 paragraphs (16.562 mean paragraphs per file)
- 14,251 sentences (1.921 mean sentences per paragraph).

By Category:

- books: 844 paragraphs and 2,030 sentences
- cinema: 1,475 paragraphs and 3,047 sentences
- cooking: 1,190 paragraphs and 2,425 sentences
- gaming: 1,802 paragraphs and 3,373 sentences
- sports: 2,109 paragraphs and 3,376 sentences

Words and Vocabulary
~~~~~~~~~~~~~~~~~~~~

Word count of 288,520 with a vocabulary of 23,738 (12.154 lexical diversity).

- books: 41,851 words with a vocabulary size of 7,838
- cinema: 69,153 words with a vocabulary size of 10,274
- cooking: 37,854 words with a vocabulary size of 5,038
- gaming: 70,778 words with a vocabulary size of 9,120
- sports: 68,884 words with a vocabulary size of 8,028

Example
-------

The hobbies corpus loader returns a ``Corpus`` object with the raw text associated with the data set. This must be vectorized into a numeric form for use with scikit-learn. For example, you could use the :class:`sklearn.feature_extraction.text.TfidfVectorizer` as follows:

.. code:: python

    from yellowbrick.datasets import load_hobbies

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split as tts

    corpus = load_hobbies()
    X = TfidfVectorizer().fit_transform(corpus.data)
    y = LabelEncoder().fit_transform(corpus.target)

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)

    model = MultinomialNB().fit(X_train, y_train)
    model.score(X_test, y_test)

For more detail on text analytics and machine learning with scikit-learn, please refer to `"Working with Text Data" <https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html>`_ in the scikit-learn documentation.

Citation
--------

Exported from S3 on: Jan 21, 2017 at 06:42.

Bengfort, Benjamin, Rebecca Bilbro, and Tony Ojeda. Applied Text Analysis with Python: Enabling Language-aware Data Products with Machine Learning. " O'Reilly Media, Inc.", 2018.

Loader
------

.. autofunction:: yellowbrick.datasets.loaders.load_hobbies
