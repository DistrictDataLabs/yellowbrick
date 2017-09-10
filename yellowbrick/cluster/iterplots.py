# yellowbrick.cluster.silhouette
# Implements visualizers using the silhouette metric for cluster evaluation.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Mon Mar 27 10:09:24 2017 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: silhouette.py [57b563b] benjamin@bengfort.com $

"""
Implements an iterative labeling visualizer for unsupervised clustering algorithms.
"""

##########################################################################
## Imports
##########################################################################

from collections import namedtuple

import numpy as np
from numpy.lib.recfunctions import append_fields
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.utils import as_float_array
from .base import ClusteringScoreVisualizer
from ..exceptions import YellowbrickValueError
from ..style import resolve_colors, color_palette
from ..utils import is_dataframe, is_structured_array, has_ndarray_int_columns

## Packages for export
# __all__ = [
#     "IterLabelsVisualizer"
# ]

class IterLabelsVisualizer(ClusteringScoreVisualizer):
    """
    TODO: Document this class!
    """

    def __init__(self, model, ax=None, group_label=None, **kwargs):
        super(IterLabelsVisualizer, self).__init__(model, ax=ax, **kwargs)

        # Visual Properties
        # TODO: Fix the color handling
        self.colormap = kwargs.get('colormap', 'set1')
        self.color = kwargs.get('color', None)
        self.group_label_ = group_label if group_label else 'group_label'

        # Required internal properties
        self.X = None
        self.y = None
        self.label_slices = None
        self.labels_ = None

    def get_label_slices(self, label_array, **kargs):
        """
        """
        # Grap the label class  column and sort it
        # Determine the what label class ids and numberic ids should be attached
        label_ids = np.unique(label_array)
        time_labels = np.arange(label_ids.size)

        # create an index to hold where labels should be applied and then apply them
        label_map = {label: position+1 for position, label in enumerate(label_ids)}
        array_labeled = [label_map[val] for val in label_array]

        # Measure the time slices for each
        self.label_slices = ndimage.find_objects(array_labeled)

        return self.label_slices

    def transform(self, X, y=None, label_col=None, label_value=None, label_array=None, **kargs):

        if label_col is None:
            raise Exception("The tranform method requires a label_col")

        if label_array is not None:
            # if the label array is present then skip the other steps
            pass

        elif isinstance(X, np.ndarray) and isinstance(label_col, int):
            label_array = X[:, label_col]

        elif is_dataframe(X):
            label_array = X[label_col].values
            X = X.as_matrix()

        # handle numpy named/ structured array
        elif is_structured_array(X):
            label_array = X[label_col]

        else:
            raise Exception("X is not a recognized data type")

        if label_value is not None:
            if label_value not in np.unique(label_array):
                raise Exception("label_value not found in the label_col")

            label_indices = np.where(label_array == label_value)
            self.X = X[label_indices]
            if y:
                self.y = y[label_indices]

        else:
            label_indices = np.argsort(label_array)
            self.X = X[label_indices]
            if y:
                self.y = y[label_indices]
            self.get_label_slices(np.sort(label_array))


    def fit_model(self, X_slice):
        clf = self.estimator()
        clf.fit(X_slice)

        labels = clf.labels_

        # Apply consistent labels
        float_array = as_float_array(X_slice, copy=True)

        ordering = []
        GroupMean = namedtuple('GroupOrder', ['cluster_group', 'group_mean'])

        for cluster_group in np.unique(labels):
            cluster_label_rows = np.where(labels==cluster_group)
            row_values = float_array[cluster_label_rows].ravel()
            ordering.append(GroupMean(cluster_group, np.mean(row_values)))

        # sort the labels by mean distance
        sorted_ordering = sorted(ordering, key=lambda x: x.group_mean)
        labels_map = {elem.cluster_group:rank for rank, elem in enumerate(sorted_ordering) }
        map_function = np.vectorize(lambda x: labels_map[x])
        mapped_labels = map_function(labels)
        return mapped_labels

    def fit(self, X, y, **kwargs):
        nrows, ncols = X.shape

        y_indices = np.argsort(y)
        self.X = X[y_indices]
        self.y = np.sort(y)

        time_slices = self.get_label_slices()

        # create memory array to hold the labels
        self.labels_ = np.empty(nrows)
        self.labels_.fill(-1)

        for t_slice in time_slices:
            X_slice = self.X[t_slice]
            # FIT TRANSFORM PREDICT
            self.labels_[t_slice] = self.fit_model(X_slice)

        # self.draw()
        return self

    def draw(self):
        labels_unique = np.unique(self.labels_)

        for time_slice in self.time_slices:
            slice_name = np.unique(self.y[time_slice])
            x_slice = self.X[time_slice]
            labels_slice = self.labels_[time_slice]

            fig, ax = self.ax.subplots()

            for label in labels_unique:
                points_indices = np.where(label==labels_slice)
                point_values = labels_slice[points_indices]
                ax.plot(point_values[:,0], point_values[:,1], marker='o', linestyle='', ms=5, label=label)

            ax.set_aspect('equal', 'datalim')
            ax.legend()
            self.plt.grid(True)
            self.plt.title("Time Series {label}".format(label=slice_name))
