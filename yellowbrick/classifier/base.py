from ..exceptions import YellowbrickTypeError
from ..utils import isclassifier
from ..base import ScoreVisualizer

import numpy as np

    
class ClassificationScoreVisualizer(ScoreVisualizer):

    def __init__(self, model, ax=None, **kwargs):
        """
        Check to see if model is an instance of a classifer.
        Should return an error if it isn't.
        """
        if not isclassifier(model):
            raise YellowbrickTypeError(
                "This estimator is not a classifier; try a regression or clustering score visualizer instead!"
        )

        super(ClassificationScoreVisualizer, self).__init__(model, ax=ax, **kwargs)

    #TODO during refactoring this can be used to generalize ClassBalance
    def class_counts(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))
