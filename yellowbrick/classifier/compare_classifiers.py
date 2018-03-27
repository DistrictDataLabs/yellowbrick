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
            for idx, c in enumerate(self.classes_):
                if idx == 0:
                    title = score_type.capitalize()
                else:
                    title = None

                visualizers.append(BarVisualizer(label=c.capitalize(), title=title))
                # TODO also append for recall etc.

        super().__init__(visualizers = visualizers, sharex = True, **kwargs)

        self.SUBPLOT_DEFAULT_WIDTH_PIXELS = 150 # Overriden in append. TODO make a calcualted property
        self.SUBPLOT_DEFAULT_HEIGHT_PIXELS = 70

    def fit(self):
        print("TODO implement fit that fits all estimeators on new data")

    def append(self, estimators):

        #TODO check on python 2 compatibility - in 2 it is iteritems()
        for key, value in estimators.items():
            self.estimators[key] = {"estimator":value,"scores":None}

        # TODO get smarter about this - scale down as we add more? Set a mix width? Do this in an @property
        pixels_per_scenario = 40
        self.SUBPLOT_DEFAULT_WIDTH_PIXELS = len(self.estimators) * pixels_per_scenario

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

                self.visualizers[visualizer_idx].fit(values,labels)
                visualizer_idx += 1        

    def finalize(self):
        self.axarr[-1,0].set_xlabel('Scenarios')
        self.axarr[-1,0].set_xticklabels(self.estimators.keys(), rotation='vertical')
        plt.subplots_adjust(hspace=0.25)

##########################################################################
## Random Visualizer
##########################################################################

class BarVisualizer(Visualizer):
    """
    
    """

    def __init__(self, ax=None, label= None, title=None, **kwargs):
       
        self.label = label
        
        super(BarVisualizer, self).__init__(ax=ax, title=title, **kwargs)
        

    def fit(self, values, labels):

        if labels is not None:
            self.labels = np.array(labels)

        self.values = values
        self.draw()

        return self

    def draw(self):
        
        self.bar_width = 0.75
        self.indices = np.arange(len(self.labels))
        self.rects = self.ax.bar(self.indices, self.values, self.bar_width)

        for rect in self.rects:
            height = rect.get_height()

            formatted_value = "{:.2g}".format(height)
            self.ax.text(
                rect.get_x() + rect.get_width()/2., 
                0.8*height,
                formatted_value,
                ha='center',
                va='bottom',
                size=8,
                color=(1,1,1,1) # TODO use the find inverse color helper
            )


        self.ax.set_ylim(0,1)
        self.ax.set_yticks([])

        return self.ax

    def finalize(self):
        if self.label != None:
            self.ax.set_ylabel(self.label)

        if self.title != None:
            # TODO this title spacing is not very good
            print(self.ax.margins(y=0.5))
            self.ax.set_title(self.title, loc='left', fontsize=12, y=0.95)

        self.ax.set_xticks(self.indices + self.bar_width / 2)
        

        return self.ax