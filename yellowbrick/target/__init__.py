# yellowbrick.target
# Implements visualizers related to the dependent (target) variable, y.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Thu Jul 19 08:57:05 2018 -0400
#
# ID: __init__.py [d742c57] benjamin@bengfort.com $

"""
Implements visualizers related to the dependent (target) variable, y. For
example, the ClassBalance visualizer shows how many of each class are
represented in the target. Other utilities include detection of sequential vs
discrete classes, binarization and thresholding visualization, and feature
correlation visualizations.
"""

##########################################################################
## Imports
##########################################################################

# Hoist visualizers into the top level of the target package
from .class_balance import ClassBalance, class_balance
from .binning import BalancedBinningReference, balanced_binning_reference
from .feature_correlation import FeatureCorrelation

# Alias the TargetType defined in yellowbrick.utils.target
from yellowbrick.utils.target import TargetType
