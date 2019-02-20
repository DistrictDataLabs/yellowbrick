.. -*- mode: rst -*-

PosTag Visualization
====================

Parts of speech (e.g. verbs, nouns, prepositions, adjectives) indicate how a word is functioning within the context of a sentence. In English as in many other languages, a single word can function in multiple ways. Part-of-speech tagging lets us encode information not only about a word’s definition, but also its use in context (for example the words “ship” and “shop” can be either a verb or a noun, depending on the context). 

The ``PosTagVisualizer`` is intended to support grammar-based feature extraction techniques for machine learning workflows that require natural language processing. The visualizer reads in a corpus that has already been sentence- and word-segmented, and tagged, creating a bar chart to visualize the relative proportions of different parts-of-speech in a corpus.

.. note::
    The ``PosTagVisualizer`` currently works with both Penn-Treebank (e.g. via NLTK) and Universal Dependencies (e.g. via SpaCy)-tagged corpora, but expects corpora that have already been tagged, and which take the form of a list of (document) lists of (sentence) lists of ``(token, tag)`` tuples, as in the example below.

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

    # Create the visualizer, fit, score, and poof it
    viz = PosTagVisualizer()
    viz.fit(tagged_stanzas)
    viz.poof()


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


API Reference
-------------

.. automodule:: yellowbrick.text.postag
    :members: PosTagVisualizer
    :undoc-members:
    :show-inheritance: