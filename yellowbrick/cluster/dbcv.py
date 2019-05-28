# yellowbrick.cluster.dbcv
# Implements Density-Based Clustering Validation score for DBSCAN
#
# Author:   Luke Wileczek <lwileczek@protonmail.com>
# Created:  Sat May 4 10:20:38 2019
#
# Copyright (C) 2019 District Data Labs
# For license information, see LICENSE.txt
#
# ID: dbcv.py

"""
Implimenting Density-Based Clustering Validation
http://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
# from .base import ClusteringScoreVisualizer
# from ..style.palettes import LINE_COLOR
# from ..exceptions import YellowbrickValueError, YellowbrickWarning

## Packages for export
__all__ = [
    "dbcv"
]

##########################################################################
## Functions
##########################################################################

"""
Start witht the eight definitions in the original paper
"""

def core_distance(X, labels, dist_func="euclidean"):
    """
    The inverse of the density of each object with respect to all other objects
    inside its cluster

    Parametes
    ---------
    X:  (array) Matrix of floats, The distances are calculated for each
        point in a cluster, not against all other points. 

    labels: (array) list of labels indicating to which cluster each record
        belongs e.g. sklearn.cluster.DBSCAN.fit().labels_

    dist_func: (string) string indicating the distance function to be used in

    OUTPUT:
        core_distances - (array) list of the core distance value for each point
        with respect to each other point in its cluster.  If it is a noise
        point, the value is zero. 

    assertions: 
    input matrix is a numpy array:  
    https://stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
    """
    assert isinstance(X, np.ndarray), "X must be a numpy array, np.ndarray"
    assert len(X) == len(labels), ("Must have a label for each point, -1 for"
        "noise")

    n_rows, n_cols = X.shape
    core_distances = np.zeros(n_rows)
    if (len(set(labels)) - (1 if -1 in labels else 0)) == 0:
        return core_distances
    clusters = set(labels)
    if -1 in clusters:
        clusters.remove(-1)  # -1 stands for noise
    for clsr in clusters:
        bln = labels == clsr
        cluster = X[bln]
        n = sum(bln) - 1 
        core_dists = np.zeros(len(cluster))
        cluster_dists = cdist(cluster, cluster, dist_func)
        with np.errstate(divide='ignore'):

            cluster_dists = cluster_dists**(-n_cols)
        cluster_dists[cluster_dists == np.inf] = 0
        core_distances[bln] = (np.sum(cluster_dists, axis=1)/n) ** (-1/n_cols)

    return core_distances


def mutual_reachability(X, core_dists, dist_func="euclidean"):
    """
    The mutual reachability between two objects

    Parameters
    ----------

    X:  (array) data matrix

    core_distances:  (array) The core distances for each row/object 

    dist_func:  (string) what distance metric to use in cdist
    """

    # the following cdist call is the slowest part of this function
    # it computes a symmetirc matrix, but perhaps if there is a better way
    # to use it and compute the lower triangle it would save time
    # Might not be worth it, hard to tell.
    dist_matrix = np.tril(cdist(X, X, dist_func))
    n_row, n_col = dist_matrix.shape

    # compute sparse matrix format
    length = n_row * (n_row-1) // 2
    dimensions = np.zeros((2, length), dtype=int)
    entries = np.zeros((3, length))
    for n in range(1, n_row):
        start = (n * (n+1) // 2)-n
        end = ((n+1) * (n+2) // 2)-n-1
        dimensions[1, start:end] = list(range(n))  # cols
        dimensions[0, start:end] = n               # rows
        entries[0, start:end] = dist_matrix[n, :n]
        entries[1, start:end] = core_dists[n]
        entries[2, start:end] = core_dists[range(n)]

    reachability = np.max(entries, axis = 0)
    reachability_graph = coo_matrix( (reachability, (dimensions[0, :],
        dimensions[1, :]))) 

    return reachability_graph
    

def cluster_density_sparseness(X):
    return None

def cluster_density_separation(X):
    return None

def cluster_validity_index(X):
    return None

def clustering_validity_index(X):
    return None

def dbcv(X, labels, distance_function="euclidean"):
    """
    Density-Based Cluster Validation
    """

    distances = core_distance(X, labels, dist_func=distance_function)
    graph = mutual_reachability(X, distances, dist_func=distance_function)
    mst = minimum_spanning_tree(graph)  # need it to be symmetric?
    return None

if __name__ == "__main__":
    """
    Testing to make sure functions work as intended

    Further testing in d_test.py file. This is mostly to ensure small
    functionallity and checking along the way. 
    """
    from sklearn.cluster import DBSCAN

    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    clusters = set(clustering.labels_)
    if -1 in clusters:
        clusters.remove(-1)  # -1 is noise not a cluster
    print(clustering.labels_)
    print(1 == clustering.labels_)
    print(X)
    print("shape:", X.shape)
    print(X[1 == clustering.labels_])
    print(clustering)
    cd_output = core_distance(X, clustering.labels_)
    print(cd_output)
    graph = mutual_reachability(X, cd_output, 'euclidean')
    print("graph:\n", graph)

