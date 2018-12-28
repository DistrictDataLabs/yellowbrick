# ID: umap.py [] jchealy@gmail.com $

"""
Generate figures for UMAP documentation.
"""

##########################################################################
## Imports
##########################################################################

import matplotlib.pyplot as plt

from corpus import load_corpus
from yellowbrick.text import UMAPVisualizer

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


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
    visualizer.poof(outpath=outpath)


##########################################################################
## Main Method
##########################################################################

if __name__ == '__main__':

    # Load and vectorize the corpus
    corpus = load_corpus("../../../data/hobbies")
    tfidf = TfidfVectorizer()

    docs   = tfidf.fit_transform(corpus.data)
    target = corpus.target

    # Whole corpus visualization
    umap(docs, target, "images/umap_all_docs_euclidean.png")

    # Whole corpus visualization
    umap(docs, target, "images/umap_all_docs_cosine.png", metric='cosine')

    # No labels
    umap(docs, None, "images/umap_no_labels.png", labels=["documents"], metric='cosine')

    # Apply clustering instead of class names.
    clusters = KMeans(n_clusters=5)
    clusters.fit(docs)

    centers = ["c{}".format(c) for c in clusters.labels_]
    umap(docs, centers, "images/umap_kmeans.png")
