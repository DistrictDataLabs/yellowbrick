.. -*- mode: rst -*-

PosTag Visualization
====================

Parts of speech (e.g. verbs, nouns, prepositions, adjectives) indicate how a word is functioning within the context of a sentence. In English as in many other languages, a single word can function in multiple ways. Part-of-speech tagging lets us encode information not only about a word’s definition, but also its use in context (for example the words “ship” and “shop” can be either a verb or a noun, depending on the context).

The ``PosTagVisualizer`` is intended to support grammar-based feature extraction techniques for machine learning workflows that require natural language processing. The visualizer can either read in a corpus that has already been sentence- and word-segmented, and tagged, or perform this tagging automatically by specifying the parser to use (nltk or spacy). The visualizer creates a bar chart to visualize the relative proportions of different parts-of-speech in a corpus.

=================   =================
Visualizer           :class:`~yellowbrick.text.postag.PosTagVisualizer`
Quick Method         :func:`~yellowbrick.text.postag.postag`
Models               Classification, Regression
Workflow             Feature Engineering
=================   =================

.. note::
    The ``PosTagVisualizer`` currently works with both Penn-Treebank (e.g. via NLTK) and Universal Dependencies (e.g. via SpaCy)-tagged corpora. This expects either raw text, or corpora that have already been tagged which take the form of a list of (document) lists of (sentence) lists of ``(token, tag)`` tuples, as in the example below.

Penn Treebank Tags
------------------

.. plot::
    :context: close-figs
    :alt: PosTag plot with Penn Treebank tags

    from yellowbrick.text import PosTagVisualizer


    tagged_stanzas = [
        [
            [
                ('Whose', 'JJ'),('woods', 'NNS'),('these', 'DT'),
                ('are', 'VBP'),('I', 'PRP'),('think', 'VBP'),('I', 'PRP'),
                ('know', 'VBP'),('.', '.')
                ],
            [
                ('His', 'PRP$'),('house', 'NN'),('is', 'VBZ'),('in', 'IN'),
                ('the', 'DT'),('village', 'NN'),('though', 'IN'),(';', ':'),
                ('He', 'PRP'),('will', 'MD'),('not', 'RB'),('see', 'VB'),
                ('me', 'PRP'),('stopping', 'VBG'), ('here', 'RB'),('To', 'TO'),
                ('watch', 'VB'),('his', 'PRP$'),('woods', 'NNS'),('fill', 'VB'),
                ('up', 'RP'),('with', 'IN'),('snow', 'NNS'),('.', '.')
                ]
            ],
        [
            [
                ('My', 'PRP$'),('little', 'JJ'),('horse', 'NN'),('must', 'MD'),
                ('think', 'VB'),('it', 'PRP'),('queer', 'JJR'),('To', 'TO'),
                ('stop', 'VB'),('without', 'IN'),('a', 'DT'),('farmhouse', 'NN'),
                ('near', 'IN'),('Between', 'NNP'),('the', 'DT'),('woods', 'NNS'),
                ('and', 'CC'),('frozen', 'JJ'),('lake', 'VB'),('The', 'DT'),
                ('darkest', 'JJS'),('evening', 'NN'),('of', 'IN'),('the', 'DT'),
                ('year', 'NN'),('.', '.')
                ]
            ],
        [
            [
                ('He', 'PRP'),('gives', 'VBZ'),('his', 'PRP$'),('harness', 'NN'),
                ('bells', 'VBZ'),('a', 'DT'),('shake', 'NN'),('To', 'TO'),
                ('ask', 'VB'),('if', 'IN'),('there', 'EX'),('is', 'VBZ'),
                ('some', 'DT'),('mistake', 'NN'),('.', '.')
                ],
            [
                ('The', 'DT'),('only', 'JJ'),('other', 'JJ'),('sound', 'NN'),
                ('’', 'NNP'),('s', 'VBZ'),('the', 'DT'),('sweep', 'NN'),
                ('Of', 'IN'),('easy', 'JJ'),('wind', 'NN'),('and', 'CC'),
                ('downy', 'JJ'),('flake', 'NN'),('.', '.')
                ]
            ],
        [
            [
                ('The', 'DT'),('woods', 'NNS'),('are', 'VBP'),('lovely', 'RB'),
                (',', ','),('dark', 'JJ'),('and', 'CC'),('deep', 'JJ'),(',', ','),
                ('But', 'CC'),('I', 'PRP'),('have', 'VBP'),('promises', 'NNS'),
                ('to', 'TO'),('keep', 'VB'),(',', ','),('And', 'CC'),('miles', 'NNS'),
                ('to', 'TO'),('go', 'VB'),('before', 'IN'),('I', 'PRP'),
                ('sleep', 'VBP'),(',', ','),('And', 'CC'),('miles', 'NNS'),
                ('to', 'TO'),('go', 'VB'),('before', 'IN'),('I', 'PRP'),
                ('sleep', 'VBP'),('.', '.')
                ]
        ]
    ]

    # Create the visualizer, fit, score, and show it
    viz = PosTagVisualizer()
    viz.fit(tagged_stanzas)
    viz.show()

Universal Dependencies Tags
---------------------------

Libraries like SpaCy use tags from the Universal Dependencies (UD) framework. The ``PosTagVisualizer`` can also be used with text tagged using this framework by specifying the ``tagset`` keyword as "universal" on instantiation.

.. plot::

    from yellowbrick.text import PosTagVisualizer

    tagged_speech = [
        [
            [
                ('In', 'ADP'),('all', 'DET'),('honesty', 'NOUN'),(',', 'PUNCT'),
                ('I', 'PRON'),('said', 'VERB'),('yes', 'INTJ'),('to', 'ADP'),
                ('the', 'DET'),('fear', 'NOUN'),('of', 'ADP'),('being', 'VERB'),
                ('on', 'ADP'),('this', 'DET'),('stage', 'NOUN'),('tonight', 'NOUN'),
                ('because', 'ADP'),('I', 'PRON'),('wanted', 'VERB'),('to', 'PART'),
                ('be', 'VERB'),('here', 'ADV'),(',', 'PUNCT'),('to', 'PART'),
                ('look', 'VERB'),('out', 'PART'),('into', 'ADP'),('this', 'DET'),
                ('audience', 'NOUN'),(',', 'PUNCT'),('and', 'CCONJ'),
                ('witness', 'VERB'),('this', 'DET'),('moment', 'NOUN'),('of', 'ADP'),
                ('change', 'NOUN')
                ],
            [
                ('and', 'CCONJ'),('I', 'PRON'),("'m", 'VERB'),('not', 'ADV'),
                ('fooling', 'VERB'),('myself', 'PRON'),('.', 'PUNCT')
                ],
            [
                ('I', 'PRON'),("'m", 'VERB'),('not', 'ADV'),('fooling', 'VERB'),
                ('myself', 'PRON'),('.', 'PUNCT')
                ],
            [
                ('Next', 'ADJ'),('year', 'NOUN'),('could', 'VERB'),('be', 'VERB'),
                ('different', 'ADJ'),('.', 'PUNCT')
                ],
            [
                ('It', 'PRON'),('probably', 'ADV'),('will', 'VERB'),('be', 'VERB'),
                (',', 'PUNCT'),('but', 'CCONJ'),('right', 'ADV'),('now', 'ADV'),
                ('this', 'DET'),('moment', 'NOUN'),('is', 'VERB'),('real', 'ADJ'),
                ('.', 'PUNCT')
                ],
            [
                ('Trust', 'VERB'),('me', 'PRON'),(',', 'PUNCT'),('it', 'PRON'),
                ('is', 'VERB'),('real', 'ADJ'),('because', 'ADP'),('I', 'PRON'),
                ('see', 'VERB'),('you', 'PRON')
                ],
            [
                ('and', 'CCONJ'), ('I', 'PRON'), ('see', 'VERB'), ('you', 'PRON')
                ],
            [
                ('—', 'PUNCT')
                ],
            [
                ('all', 'ADJ'),('these', 'DET'),('faces', 'NOUN'),('of', 'ADP'),
                ('change', 'NOUN')
                ],
            [
                ('—', 'PUNCT'),('and', 'CCONJ'),('now', 'ADV'),('so', 'ADV'),
                ('will', 'VERB'),('everyone', 'NOUN'),('else', 'ADV'), ('.', 'PUNCT')
                ]
        ]
    ]

    # Create the visualizer, fit, score, and show it
    viz = PosTagVisualizer(tagset="universal")
    viz.fit(tagged_speech)
    viz.show()


Quick Method
------------

The same functionality above can be achieved with the associated quick method ``postag``. This method will build the ``PosTagVisualizer`` object with the associated arguments, fit it, then (optionally) immediately show the visualization.

.. plot::
    :context: close-figs
    :alt: postag quick method with Penn Treebank tags

    from yellowbrick.text.postag import postag

    machado = [
        [
            [
                ('Last', 'JJ'), ('night', 'NN'), ('as', 'IN'), ('I', 'PRP'),
                ('was', 'VBD'), ('sleeping', 'VBG'), (',', ','), ('I', 'PRP'),
                ('dreamt', 'VBP'), ('—', 'RB'), ('marvelous', 'JJ'), ('error', 'NN'),
                ('!—', 'IN'), ('that', 'DT'), ('I', 'PRP'), ('had', 'VBD'), ('a', 'DT'),
                ('beehive', 'NN'), ('here', 'RB'), ('inside', 'IN'), ('my', 'PRP$'),
                ('heart', 'NN'), ('.', '.')
                ],
            [
                ('And', 'CC'), ('the', 'DT'), ('golden', 'JJ'), ('bees', 'NNS'),
                ('were', 'VBD'), ('making', 'VBG'), ('white', 'JJ'), ('combs', 'NNS'),
                ('and', 'CC'), ('sweet', 'JJ'), ('honey', 'NN'), ('from', 'IN'),
                ('my', 'PRP$'), ('old', 'JJ'), ('failures', 'NNS'), ('.', '.')
                ]
        ]
    ]

    # Create the visualizer, fit, score, and show it
    postag(machado)


Part of Speech Tags
-------------------

+-------------------+------------------------------------------+----------------------+--------------------------+
| Penn-Treebank Tag | Description                              | Universal Tag        | Description              |
+===================+==========================================+======================+==========================+
| CC                | Coordinating conjunction                 | ADJ                  | adjective                |
+-------------------+------------------------------------------+----------------------+--------------------------+
| CD                | Cardinal number                          | ADP                  | adposition               |
+-------------------+------------------------------------------+----------------------+--------------------------+
| DT                | Determiner                               | ADV                  | adverb                   |
+-------------------+------------------------------------------+----------------------+--------------------------+
| EX                | Existential *there*                      | AUX                  | auxiliary                |
+-------------------+------------------------------------------+----------------------+--------------------------+
| FW                | Foreign word                             | CONJ                 | conjunction              |
+-------------------+------------------------------------------+----------------------+--------------------------+
| IN                | Preposition or subordinating conjunction | CCONJ                | coordinating conjunction |
+-------------------+------------------------------------------+----------------------+--------------------------+
| JJ                | Adjective                                |  DET                 | determiner               |
+-------------------+------------------------------------------+----------------------+--------------------------+
| JJR               | Adjective, comparative                   | INTJ                 | interjection             |
+-------------------+------------------------------------------+----------------------+--------------------------+
| JJS               | Adjective, superlative                   | NOUN                 | noun                     |
+-------------------+------------------------------------------+----------------------+--------------------------+
| LS                | List item marker                         | NUM                  | numeral                  |
+-------------------+------------------------------------------+----------------------+--------------------------+
| MD                | Modal                                    | PART                 | particle                 |
+-------------------+------------------------------------------+----------------------+--------------------------+
| NN                | Noun, singular or mass                   | PRON                 | pronoun                  |
+-------------------+------------------------------------------+----------------------+--------------------------+
| NNS               | Noun, plural                             | PROPN                | proper noun              |
+-------------------+------------------------------------------+----------------------+--------------------------+
| NNP               | Proper noun, singular                    | PUNCT                | punctuation              |
+-------------------+------------------------------------------+----------------------+--------------------------+
| NNPS              | Proper noun, plural                      | SCONJ                | subordinating conjunction|
+-------------------+------------------------------------------+----------------------+--------------------------+
| PDT               | Predeterminer                            | SYM                  | symbol                   |
+-------------------+------------------------------------------+----------------------+--------------------------+
| POS               | Possessive ending                        | VERB                 | verb                     |
+-------------------+------------------------------------------+----------------------+--------------------------+
| PRP               | Personal pronoun                         | X                    | other                    |
+-------------------+------------------------------------------+----------------------+--------------------------+
| PRP$              | Possessive pronoun                       | SPACE                | space                    |
+-------------------+------------------------------------------+----------------------+--------------------------+
| RB                | Adverb                                   |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| RBR               | Adverb, comparative                      |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| RBS               | Adverb, superlative                      |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| RP                | Particle                                 |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| SYM               | Symbol                                   |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| TO                | *to*                                     |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| UH                | Interjection                             |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| VB                | Verb, base form                          |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| VBD               | Verb, past tense                         |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| VBG               | Verb, gerund or present participle       |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| VBN               | Verb, past participle                    |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| VBP               | Verb, non-3rd person singular present    |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| VBZ               | Verb, 3rd person singular present        |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| WDT               | Wh-determiner                            |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| WP                | Wh-pronoun                               |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| WP$               | Possessive wn-pronoun                    |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+
| WRB               | Wh-adverb                                |                      |                          |
+-------------------+------------------------------------------+----------------------+--------------------------+

Parsing raw text automatically
------------------------------

The ``PosTagVisualizer`` can also be used with untagged text by using the ``parse`` keyword on instantiation. The keyword
to parse indicates which natural language processing library to use. To use ``spacy``:


.. code:: python

    untagged_speech = u'Whose woods these are I think I know'

    # Create the visualizer, fit, score, and show it
    viz = PosTagVisualizer(parser='spacy')
    viz.fit(untagged_speech)
    viz.show()


Or, using the ``nltk`` parser.

.. code:: python

    untagged_speech = u'Whose woods these are I think I know'

    # Create the visualizer, fit, score, and show it
    viz = PosTagVisualizer(parser='nltk')
    viz.fit(untagged_speech)
    viz.show()

.. note::
    To use either of these parsers, either `nltk` or `spacy` must already be installed in your environment.

You can also change the tagger used. For example, using `nltk` you can select either `word` (default):

.. code:: python

    untagged_speech = u'Whose woods these are I think I know'

    # Create the visualizer, fit, score, and show it
    viz = PosTagVisualizer(parser='nltk_word')
    viz.fit(untagged_speech)
    viz.show()

Or using `wordpunct`.

.. code:: python

    untagged_speech = u'Whose woods these are I think I know'

    # Create the visualizer, fit, score, and show it
    viz = PosTagVisualizer(parser='nltk_wordpunct')
    viz.fit(untagged_speech)
    viz.show()


API Reference
-------------

.. automodule:: yellowbrick.text.postag
    :members: PosTagVisualizer, postag
    :undoc-members:
    :show-inheritance:
