#!/usr/bin/env python3
# ID: umap_vis.py [73a44e5] jchealy@gmail.com $

"""
Manually generate figures for the UMAP documentation.
"""
##########################################################################
## Imports
##########################################################################

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from yellowbrick.text import UMAPVisualizer
from yellowbrick.datasets import load_hobbies

##########################################################################
## Generate
##########################################################################


def umap(docs, target, outpath, **kwargs):
    # Create a new figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Visualize the frequency distribution
    visualizer = UMAPVisualizer(ax=ax, **kwargs)
    visualizer.fit(docs, target)
    visualizer.show(outpath=outpath)


if __name__ == "__main__":

    # Load and vectorize the corpus
    corpus = load_hobbies()
    tfidf = TfidfVectorizer()

    docs = tfidf.fit_transform(corpus.data)
    labels = corpus.target

    # Whole corpus visualization
    umap(docs, labels, "images/umap.png")

    # Whole corpus visualization
    umap(docs, labels, "images/umap_cosine.png", metric="cosine")

    # No labels
    umap(docs, None, "images/umap_no_labels.png", labels=["documents"], metric="cosine")

    # Apply clustering instead of class names
    clusters = KMeans(n_clusters=5)
    clusters.fit(docs)

    centers = ["c{}".format(c) for c in clusters.labels_]
    umap(docs, centers, "images/umap_kmeans.png")
