import matplotlib.pyplot as plt

from corpus import load_corpus
from yellowbrick.text import TSNEVisualizer

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def tsne(docs, target, outpath, **kwargs):
    # Create a new figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Visualize the frequency distribution
    visualizer = TSNEVisualizer(ax=ax, **kwargs)
    visualizer.fit(docs, target)
    visualizer.poof(outpath=outpath)


if __name__ == '__main__':

    # Load and vectorize the corpus
    corpus = load_corpus("../../../examples/data/hobbies")
    tfidf = TfidfVectorizer()

    docs   = tfidf.fit_transform(corpus.data)
    target = corpus.target

    # Whole corpus visualization
    tsne(docs, target, "images/tsne_all_docs.png")

    # No labels
    tsne(docs, None, "images/tsne_no_labels.png", labels=["documents"])

    # Apply clustering instead of class names.
    clusters = KMeans(n_clusters=5)
    clusters.fit(docs)

    centers = ["c{}".format(c) for c in clusters.labels_]
    tsne(docs, centers, "images/tsne_kmeans.png")
