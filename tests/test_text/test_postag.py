# tests.test_text.test_postag
# Tests for the part-of-speech tagging visualization
#
# Author:   Rebecca Bilbro
# Created:  2019-02-19 21:29
#
# Copyright (C) 2019 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_postag.py [bd9cbb9] rebecca.bilbro@bytecubed.com $

"""
Tests for the part-of-speech tagging visualization
"""

##########################################################################
## Imports
##########################################################################

import pytest
import matplotlib.pyplot as plt

from tests.base import VisualTestCase
from tests.base import IS_WINDOWS_OR_CONDA

from yellowbrick.text.postag import *
from yellowbrick.exceptions import YellowbrickValueError

try:
    import nltk
    from nltk import pos_tag, sent_tokenize
    from nltk import word_tokenize, wordpunct_tokenize
except ImportError:
    nltk = None

try:
    import spacy
except ImportError:
    spacy = None


##########################################################################
## Data
##########################################################################

sonnets = [
    """
    FROM fairest creatures we desire increase,
    That thereby beauty's rose might never die,
    But as the riper should by time decease,
    His tender heir might bear his memory:
    But thou, contracted to thine own bright eyes,
    Feed'st thy light'st flame with self-substantial fuel,
    Making a famine where abundance lies,
    Thyself thy foe, to thy sweet self too cruel.
    Thou that art now the world's fresh ornament
    And only herald to the gaudy spring,
    Within thine own bud buriest thy content
    And, tender churl, makest waste in niggarding.
    Pity the world, or else this glutton be,
    To eat the world's due, by the grave and thee.
    """,
    """
    When forty winters shall beseige thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery, so gazed on now,
    Will be a tatter'd weed, of small worth held:
    Then being ask'd where all thy beauty lies,
    Where all the treasure of thy lusty days,
    To say, within thine own deep-sunken eyes,
    Were an all-eating shame and thriftless praise.
    How much more praise deserved thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.
    """,
    """
    Look in thy glass, and tell the face thou viewest
    Now is the time that face should form another;
    Whose fresh repair if now thou not renewest,
    Thou dost beguile the world, unbless some mother.
    For where is she so fair whose unear'd womb
    Disdains the tillage of thy husbandry?
    Or who is he so fond will be the tomb
    Of his self-love, to stop posterity?
    Thou art thy mother's glass, and she in thee
    Calls back the lovely April of her prime:
    So thou through windows of thine age shall see
    Despite of wrinkles this thy golden time.
    But if thou live, remember'd not to be,
    Die single, and thine image dies with thee.
    """,
]


##########################################################################
## PosTag Utils
##########################################################################


def check_nltk_data():
    """
    Returns True if NLTK data has been downloaded, False otherwise
    """
    try:
        nltk.data.find("corpora/treebank")
        return True
    except LookupError:
        pytest.xfail("error occured because nltk postag data is not available")


def check_spacy_data():
    """
    Returns True if SpaCy data has been downloaded, False otherwise
    """
    try:
        spacy.load("en_core_web_sm")
        return True
    except OSError:
        pytest.xfail("error occured because spacy data model is not available")


def get_tagged_docs(X, model="nltk", tagger="word"):
    """
    X is a list of strings; each string is a single document.
    For each document, perform part-of-speech tagging, and
    yield a list of sentences, where each sentence is a list
    of (token, tag) tuples

    If model=="nltk", `NLTK` will be used to sentence and word
    tokenize the incoming documents. User may select the `NLTK`
    tagger to be used; (for now) either the word tokenizer or the
    workpunct tokenizer.

    If model=="spacy", `SpaCy` will be used to sentence and word
    tokenize the incoming documents.
    """
    if model == "spacy":
        nlp = spacy.load("en_core_web_sm")
        for doc in X:
            tagged = nlp(doc)
            yield [
                list((token.text, token.pos_) for token in sent)
                for sent in tagged.sents
            ]

    elif model == "nltk":
        if tagger == "wordpunct":
            for doc in X:
                yield [pos_tag(wordpunct_tokenize(sent)) for sent in sent_tokenize(doc)]
        else:
            for doc in X:
                yield [pos_tag(word_tokenize(sent)) for sent in sent_tokenize(doc)]


##########################################################################
## PosTag Tests
##########################################################################


class TestPosTag(VisualTestCase):
    """
    PosTag (Part of Speech Tagging Visualizer) Tests
    """

    def test_quick_method(self):
        """
        Assert no errors occur when using the quick method
        """
        # Fail if data hasn't been downloaded
        check_nltk_data()

        _, ax = plt.subplots()
        tagged_docs = list(get_tagged_docs(sonnets))

        viz = postag(tagged_docs, ax=ax, show=False)
        viz.ax.grid(False)

        # Fails on Miniconda/Appveyor with images not close (RMS 5.157)
        tol = 5.5 if IS_WINDOWS_OR_CONDA else 0.25
        self.assert_images_similar(viz, tol=tol)

    def test_unknown_tagset(self):
        """
        Ensure an exception is raised if the specified tagset is unknown
        """
        with pytest.raises(YellowbrickValueError):
            PosTagVisualizer(tagset="brill")

    def test_frequency_mode(self):
        """
        Assert no errors occur when the visualizer is run on frequency mode
        """
        check_nltk_data()

        _, ax = plt.subplots()
        tagged_docs = list(get_tagged_docs(sonnets))

        viz = PosTagVisualizer(ax=ax, frequency=True)
        viz.fit(tagged_docs)
        viz.finalize()
        ax.grid(False)

        # Sorted tags i.e predetermined order
        sorted_tags = [
            "noun",
            "adjective",
            "punctuation",
            "verb",
            "preposition",
            "determiner",
            "adverb",
            "conjunction",
            "pronoun",
            "wh- word",
            "modal",
            "infinitive",
            "possessive",
            "other",
            "symbol",
            "existential",
            "digit",
            "non-English",
            "interjection",
            "list",
        ]

        # Extract tick labels from the plot
        ticks_ax = [tick.get_text() for tick in ax.xaxis.get_ticklabels()]

        # Assert that ticks are set properly
        assert ticks_ax == sorted_tags

        # Fails on Miniconda/Appveyor with images not close (RMS 5.302)
        tol = 5.5 if IS_WINDOWS_OR_CONDA else 0.5
        self.assert_images_similar(ax=ax, tol=tol)

    @pytest.mark.skipif(nltk is None, reason="test requires nltk")
    def test_word_tagged(self):
        """
        Assert no errors occur during PosTagVisualizer integration
        with word tokenized corpus
        """
        # Fail if data hasn't been downloaded
        check_nltk_data()

        tagged_docs = list(get_tagged_docs(sonnets, model="nltk", tagger="word"))

        visualizer = PosTagVisualizer(tagset="penn_treebank")

        visualizer.fit(tagged_docs)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(nltk is None, reason="test requires nltk")
    def test_wordpunct_tagged(self):
        """
        Assert no errors occur during PosTagVisualizer integration
        with wordpunct tokenized corpus
        """
        # Fail if data hasn't been downloaded
        check_nltk_data()

        wordpunct_tagged_docs = list(
            get_tagged_docs(sonnets, model="nltk", tagger="wordpunct")
        )

        visualizer = PosTagVisualizer(tagset="penn_treebank")

        visualizer.fit(wordpunct_tagged_docs)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(spacy is None, reason="test requires spacy")
    def test_spacy_tagged(self):
        """
        Assert no errors occur during PosTagVisualizer integration
        with spacy tokenized corpus
        """
        # Fail if data hasn't been downloaded
        check_spacy_data()

        spacy_tagged_docs = list(get_tagged_docs(sonnets, model="spacy"))

        visualizer = PosTagVisualizer(tagset="universal")

        visualizer.fit(spacy_tagged_docs)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(spacy is None, reason="test requires spacy")
    def test_spacy_raw(self):
        """
        Assert no errors occur during PosTagVisualizer integration
        with raw corpus to be parsed using spacy
        """
        visualizer = PosTagVisualizer(parser='spacy', tagset='universal')
        visualizer.fit(sonnets)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(nltk is None, reason="test requires nltk")
    def test_nltk_word_raw(self):
        """
        Assert no errors occur during PosTagVisualizer integration
        with raw corpus to be parsed using nltk
        """
        visualizer = PosTagVisualizer(parser='nltk', tagset="penn_treebank")
        visualizer.fit(sonnets)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(nltk is None, reason="test requires nltk")
    def test_nltk_wordpunct_raw(self):
        """
        Assert no errors occur during PosTagVisualizer integration
        with raw corpus to be parsed using nltk
        """
        visualizer = PosTagVisualizer(parser='nltk_wordpunct', tagset="penn_treebank")
        visualizer.fit(sonnets)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer)

    def test_stack_mode(self):
        """
        Assert no errors occur when the visualizer is run on stack mode
        """
        check_nltk_data()

        _, ax = plt.subplots()
        tagged_docs = list(get_tagged_docs(sonnets))

        visualizer = PosTagVisualizer(stack=True, ax=ax)
        visualizer.fit(tagged_docs, y=["a", "b", "c"])
        visualizer.ax.grid(False)

        self.assert_images_similar(ax=ax)

    def test_stack_frequency_mode(self):
        """
        Assert no errors occur when the visualizer is run on both stack and
        frequency mode
        """
        check_nltk_data()

        _, ax = plt.subplots()
        tagged_docs = list(get_tagged_docs(sonnets))

        visualizer = PosTagVisualizer(stack=True, frequency=True, ax=ax)
        visualizer.fit(tagged_docs, y=["a", "b", "c"])
        visualizer.ax.grid(False)

        # Sorted tags i.e predetermined order
        sorted_tags = [
            "noun",
            "adjective",
            "punctuation",
            "verb",
            "preposition",
            "determiner",
            "adverb",
            "conjunction",
            "pronoun",
            "wh- word",
            "modal",
            "infinitive",
            "possessive",
            "other",
            "symbol",
            "existential",
            "digit",
            "non-English",
            "interjection",
            "list",
        ]

        # Extract tick labels from the plot
        ticks_ax = [tick.get_text() for tick in ax.xaxis.get_ticklabels()]

        # Assert that ticks are set properly
        assert ticks_ax == sorted_tags

        self.assert_images_similar(ax=ax)
