##########################################################################
## Imports
##########################################################################
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support

from ..base import VisualizerGrid, Visualizer
from ..style.palettes import color_palette

#TODO remove this
from .classification_report import ClassificationReport
from sklearn.linear_model import LogisticRegression

SCORES_KEYS = ('precision', 'recall', 'f1')

##########################################################################
## Compare Classifiers
##########################################################################

class CompareClassifiers(VisualizerGrid):
    """

    """
    def __init__(self, classes=None, **kwargs):
    
        self.estimators = OrderedDict()

        # TODO Some duplication from classification
        # Convert to array if necessary to match estimator.classes_
        if classes is not None:
            classes = np.array(classes)

        self.classes_  = classes

        # WARNING - order of visualizers here needs to match draw()
        self.scores_to_plot = ('precision','recall')
        visualizers = []
        for score_type in self.scores_to_plot:
            for c in self.classes_:
                visualizers.append(BarVisualizer())
                # TODO also append for recall etc.

        super().__init__(visualizers = visualizers,**kwargs)

        self.SUBPLOT_DEFAULT_PIXELS = 100

    def fit(self):
        print("Test ran fit")
    def append(self, estimators):

        #TODO check on python 2 compatibility - in 2 it is iteritems()
        for key, value in estimators.items():
            self.estimators[key] = {"estimator":value,"scores":None}

    def score(self,X,y, only_new=False):
        for key, e in self.estimators.items():
            y_pred = e["estimator"].predict(X)
            scores = precision_recall_fscore_support(y, y_pred)
            #zip methods copied from classification_report
            scores = map(lambda s: dict(zip(self.classes_, s)), scores[0:3])
            e["scores"] = dict(zip(SCORES_KEYS, scores))

    def draw(self):
        # WARNING visualizer order is linked to __init__
        visualizer_idx = 0
        for score_type in self.scores_to_plot:
            for c in self.classes_:
                values, labels = [] , []
                for label, e in self.estimators.items():
                    values.append(e['scores'][score_type][c])
                    labels.append(label)
                print(c,values)
                self.visualizers[visualizer_idx].fit(values,labels)
                visualizer_idx += 1

    #def finalize(self):
    #    print("Test ran finalize")


##########################################################################
## Random Visualizer
##########################################################################

class BarVisualizer(Visualizer):
    """
    
    """

    def __init__(self, ax=None, **kwargs):
        
        super(BarVisualizer, self).__init__(ax=ax, **kwargs)
        

    def fit(self, values, labels):

        if labels is not None:
            self.labels = np.array(labels)

        self.values = values
        self.draw()

        return self

    def draw(self):
        
        indices = np.arange(len(self.labels))
        self.ax.bar(indices, self.values, 0.75)

        return self.ax

    def finalize(self):

        return self.ax