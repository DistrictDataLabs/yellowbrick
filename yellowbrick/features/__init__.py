# yellowbrick.features
# Visualizers for feature analysis and diagnostics.
#
# Author:   Benjamin Bengfort
# Created:  Mon Oct 03 21:30:18 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: __init__.py [0f4b236] benjamin@bengfort.com $

"""
Visualizers for feature analysis and diagnostics.
"""

##########################################################################
## Imports
##########################################################################

## Hoist visualizers into the features namespace
from .pcoords import ParallelCoordinates, parallel_coordinates
from .radviz import RadialVisualizer, RadViz, radviz
from .rankd import Rank1D, rank1d, Rank2D, rank2d
from .jointplot import JointPlot, JointPlotVisualizer, joint_plot
from .pca import PCA, PCADecomposition, pca_decomposition
from .manifold import Manifold, manifold_embedding

# Alias the TargetType defined in yellowbrick.utils.target
from yellowbrick.utils.target import TargetType

# RFECV and Feature Importances moved to model selection module as of YB v1.0
from yellowbrick.model_selection.rfecv import RFECV, rfecv
from yellowbrick.model_selection.importances import FeatureImportances
from yellowbrick.model_selection.importances import feature_importances
