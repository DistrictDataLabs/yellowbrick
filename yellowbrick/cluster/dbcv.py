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
# from .base import ClusteringScoreVisualizer
# from ..style.palettes import LINE_COLOR
# from ..exceptions import YellowbrickValueError, YellowbrickWarning

## Packages for export
__all__ = [
    "dbcv"
]

##########################################################################
## Metrics
##########################################################################

"""
Start witht the eight definitions in the original paper
"""
def dbcv(X):
    """
    Density-Based Cluster Validation
    """
    return None

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

    n_row, n_cols = X.shape
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

    QUESTION: could create lower triangular matrix L such that the entries are
    the mutual_reachability of for the coordinate pair. This could be hard on
    memory though.  Is it better to go element wise as needed? 

    If poinwise is better than this function call is likely unnecessary
    complicaiton and can just be a line or two inside of the graph or tree

    Parameters
    ----------

    X:  (array) data matrix
    core_distances
    p0
    p1
    dist_func
    """

    mr_output = max(core_dists[p0], core_dists[p1], 
        cdist(X[p0, :], X[p1, :], dist_func))

    if len(X) > 5000:
        # Then it's worth using sparse vectors
        dist_matrix = np.tril(cdist(X, X, dist_func))
        n_row, n_col = dist_matrix.shape
        rows = list(range(n_row))
        entries = np.zeros(n_row*(n_row)/2)
        columns = np.zeros(n_row*(n_row)/2)
        for n in rows:
            entries[(n* ]

    return mr_output

def mutual_reachability_distance_graph(X):
    """
    The graph is the matrix of the mr in each entry

    Can iterate over the matrix and check each entry or turn it to a sparse
    matrix, three vectors for entry, row, column, and comput it that way to save
    memory Perhaps have a sparse option for large matrices?
    """

    return None

def mutual_reachability_distance_MST(X):
    return None

def cluster_density_sparseness(X):
    return None

def cluster_density_separation(X):
    return None

def cluster_validity_index(X):
    return None

def clustering_validity_index(X):
    return None


if __name__ == "__main__":
    """
    Testing to make sure functions work as intended
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

