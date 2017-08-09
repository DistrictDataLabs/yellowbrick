import matplotlib.pyplot as plt

from corpus import load_corpus
from collections import defaultdict
from yellowbrick.text import TSNEVisualizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def tsne(docs, labels, outpath, **kwargs):
    # Create a new figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Visualize the frequency distribution
    visualizer = TSNEVisualizer(**kwargs)
    visualizer.fit(docs, labels)
    visualizer.poof(outpath=outpath)


if __name__ == '__main__':

    # Load and vectorize the corpus
    corpus = load_corpus("../../../examples/data/hobbies")
    tfidf = TfidfVectorizer()

    docs   = tfidf.fit_transform(corpus.data)
    labels = corpus.target

    # Whole corpus visualization
    tsne(docs, labels, "images/tsne_all_docs.png")

    # Partial corpus visualization
    # Only visualize the sports, cinema, and gaming classes
    tsne(docs, labels, "images/tsne_limit_classes.png", classes=['sports', 'cinema', 'gaming'])

    # No labels
    tsne(docs, None, "images/tsne_no_labels.png")

    # Apply clustering instead of class names.
    clusters = KMeans(n_clusters=5)
    clusters.fit(docs)

    centers = ["c{}".format(c) for c in clusters.labels_]
    tsne(docs, centers, "images/tsne_kmeans.png")
