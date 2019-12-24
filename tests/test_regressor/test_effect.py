"""
Test the EffectPlot functionality
"""

##########################################################################
## Imports
##########################################################################

import pytest

from yellowbrick.regressor.effect import *
from yellowbrick.datasets import load_concrete

from tests.base import VisualTestCase

from sklearn.linear_model import LinearRegression, Lasso, Ridge

##########################################################################
## EffectPlot Tests
##########################################################################

class TestEffectPlot(VisualTestCase):
    """
    Test the EffectPlot base class
    """

    def test_quick_method(self):
        """
        Test quick method
        """
        X, y = load_concrete()
        visualizer = effectplot(LinearRegression(), X, y, show=False)
        self.assert_images_similar(visualizer)
    
    def test_marker(self):
        """
        Test circular marker.
        """
        X, y = load_concrete()
        visualizer = EffectPlot(LinearRegression(), marker='.', colormap='cool')
        visualizer.fit(X, y)
        visualizer.finalize()
        self.assert_images_similar(visualizer)
    
    @pytest.mark.parametrize("model", [LinearRegression(), Lasso(alpha=100), Ridge()])
    def test_other_linear_models(self, model):
        """
        Test other linear_models
        """
        message = "case failed for {}".format(model)
        viz = EffectPlot(model=model, marker='.', colormap='cool')
        assert viz.estimator == model, message
            
        