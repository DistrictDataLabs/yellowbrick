"""
Regressor visualizers that score residuals: prediction vs. actual data.
"""

##########################################################################
## Imports
##########################################################################


import matplotlib.pyplot as plt

try:
    # Only available in Matplotlib >= 2.0.2
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    make_axes_locatable = None

from sklearn.model_selection import train_test_split

from .base import RegressionScoreVisualizer
from ..draw import manual_legend
from ..style.palettes import LINE_COLOR
from ..utils.decorators import memoized
from ..exceptions import YellowbrickValueError
from ..bestfit import draw_best_fit, draw_identity_line


## Packages for export
__all__ = [
    "PredictionError", "prediction_error",
    "ResidualsPlot", "residuals_plot"
]


class EffectPlot(RegressionScoreVisualizer):
    
    def __init__(self, model, ax=None, shared_limits=True,
                 bestfit=True, identity=True, alpha=0.75, **kwargs):
        # Initialize the visualizer
        super(EffectPlot, self).__init__(model, ax=ax, **kwargs)

        # Visual arguments
        self.colors = {
            'point': kwargs.pop('point_color', None),
            'line': kwargs.pop('line_color', LINE_COLOR),
        }
        # Drawing arguments
        self.shared_limits = shared_limits
        self.bestfit = bestfit
        self.identity = identity
        self.alpha = alpha
