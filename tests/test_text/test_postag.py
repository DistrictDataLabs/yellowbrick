# -*- coding: utf8 -*-
# tests.test_text.test_postag
# Tests for the part-of-speech tagging visualization
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  2017-03-22 15:46
#
# Copyright (C) 2017 District Data Labs
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

from yellowbrick.text.postag import *


try:
    import nltk
    from nltk import pos_tag, word_tokenize
except ImportError:
    nltk = None


##########################################################################
## Data
##########################################################################

pie =  """
In a small saucepan, combine sugar and eggs
until well blended. Cook over low heat, stirring
constantly, until mixture reaches 160° and coats
the back of a metal spoon. Remove from the heat.
Stir in chocolate and vanilla until smooth. Cool
to lukewarm (90°), stirring occasionally. In a small
bowl, cream butter until light and fluffy. Add cooled
chocolate mixture; beat on high speed for 5 minutes
or until light and fluffy. In another large bowl,
beat cream until it begins to thicken. Add
confectioners' sugar; beat until stiff peaks form.
Fold into chocolate mixture. Pour into crust. Chill
for at least 6 hours before serving. Garnish with
whipped cream and chocolate curls if desired.
"""


##########################################################################
## PosTag Tests
##########################################################################

class TestPosTag(object):
    """
    PosTag (Part of Speech Tagging Visualizer) Tests
    """

    @pytest.mark.skipif(nltk is None, reason="test requires nltk")
    def test_integrated_postag(self):
        """
        Assert no errors occur during postag integration
        """
        tokens = word_tokenize(pie)
        tagged = pos_tag(tokens)

        visualizer = PosTagVisualizer()
        visualizer.transform(tagged)
