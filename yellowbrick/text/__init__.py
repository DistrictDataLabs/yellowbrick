# yellowbrick.text
# Visualizers for text feature analysis and diagnostics.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  2017-01-20 14:42
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: __init__.py [75d9b20] rebecca.bilbro@bytecubed.com $

"""
Visualizers for text feature analysis and diagnostics.
"""

##########################################################################
## Imports
##########################################################################

from .tsne import TSNEVisualizer, tsne
from .freqdist import FreqDistVisualizer, freqdist
from .postag import PosTagVisualizer
