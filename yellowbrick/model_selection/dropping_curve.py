# yellowbrick.model_selection.dropping_curve
# Implements a feature dropping curve visualization for model selection.
#
# Author:   Charles Guan
# Created:  Wed Dec 8 15:03:00 2021 -0800

"""
Implements a random-input-dropout curve visualization for model selection.
Another common name: neuron dropping curve (NDC), in neural decoding research
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from yellowbrick.base import ModelVisualizer
from yellowbrick.style import resolve_colors
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.model_selection import validation_curve as sk_validation_curve


# Default ticks for the model selection curve, relative number of features
DEFAULT_FEATURE_SIZES = np.linspace(0.1, 1.0, 5)
