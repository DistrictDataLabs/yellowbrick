# Clustering Evaluation Imports
from functools import partial

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs as sk_make_blobs

from yellowbrick.cluster import InterclusterDistance

# Helpers for easy dataset creation
N_SAMPLES = 1000
N_FEATURES = 12
SHUFFLE = True

# Make blobs partial
make_blobs = partial(sk_make_blobs, n_samples=N_SAMPLES, n_features=N_FEATURES, shuffle=SHUFFLE)


if __name__ == '__main__':
    # Make 8 blobs dataset
    X, y = make_blobs(centers=12)

    # Instantiate the clustering model and visualizer
    # Instantiate the clustering model and visualizer
    visualizer = InterclusterDistance(KMeans(9))

    visualizer.fit(X) # Fit the training data to the visualizer
    visualizer.poof(outpath="images/icdm.png") # Draw/show/poof the data
