# ID: dispersion.py [] lwgray@gmail.com $

"""
Generate figures for Dispersion Plot documentation.
"""

##########################################################################
## Imports
##########################################################################

import matplotlib.pyplot as plt

from corpus import load_corpus
from yellowbrick.text.dispersion import DispersionPlot

##########################################################################
## Generate
##########################################################################

def dispersion(target_words, text, outpath, **kwargs):
    # Create a new figure and axes
    _, ax = plt.subplots()

    # Visualize the Dispersion of target words
    visualizer = DispersionPlot(target_words, ax=ax, **kwargs)
    visualizer.fit(text)
    visualizer.poof(outpath=outpath)


##########################################################################
## Main Method
##########################################################################

if __name__ == '__main__':

    # Load the corpus
    corpus = load_corpus("../../../examples/data/hobbies")

    # Convert corpus into a list of all words from beginning to end
    text = [word for doc in corpus.data for word in doc.split()]
    
    # Select target words to visualize 
    target_words = ['Game', 'player', 'score', 'oil', 'Man']

    # Display dispersion of target words throughout corpus
    dispersion(target_words, text, "images/dispersion_docs.png")


