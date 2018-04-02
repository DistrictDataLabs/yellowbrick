import matplotlib.pyplot as plt

from corpus import load_corpus
from collections import defaultdict
from yellowbrick.text.freqdist import FreqDistVisualizer
from sklearn.feature_extraction.text import CountVectorizer


def freqdist(docs, outpath, corpus_kwargs={}, **kwargs):
    # Create a new figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Vectorize the corpus
    vectorizer = CountVectorizer(**corpus_kwargs)
    docs       = vectorizer.fit_transform(docs)
    features   = vectorizer.get_feature_names()

    # Visualize the frequency distribution
    visualizer = FreqDistVisualizer(ax=ax, features=features, **kwargs)
    visualizer.fit(docs)
    visualizer.poof(outpath=outpath)


if __name__ == '__main__':

    # Load the corpus
    corpus = load_corpus("../../../examples/data/hobbies")

    # Whole corpus visualization
    freqdist(corpus.data, "images/freqdist_corpus.png", orient='v')

    # Stopwords removed
    freqdist(corpus.data, "images/freqdist_stopwords.png", {'stop_words': 'english'}, orient='v')

    # Specific categories
    hobbies = defaultdict(list)
    for text, label in zip(corpus.data, corpus.target):
        hobbies[label].append(text)

    # Cooking Category
    freqdist(hobbies["cooking"], "images/freqdist_cooking.png", {'stop_words': 'english'}, orient='v')

    # Gaming Category
    freqdist(hobbies["gaming"], "images/freqdist_gaming.png", {'stop_words': 'english'}, orient='v')
