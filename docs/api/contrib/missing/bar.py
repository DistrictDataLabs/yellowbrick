import numpy as np
from sklearn.datasets import make_classification

# Create dummy data
X, y = make_classification(
        n_samples=400, n_features=10, n_informative=2, n_redundant=3,
        n_classes=2, n_clusters_per_class=2, random_state=854
    )

# assign some NaN values
X[X > 1.5] = np.nan
features = ["Feature {}".format(str(n)) for n in range(10)]

from yellowbrick.contrib.missing import MissingValuesBar

viz = MissingValuesBar(features=features)
viz.fit(X)
viz.poof(outpath="images/missingbar.png")


viz = MissingValuesBar(features=features)
viz.fit(X, y=y)
viz.poof(outpath="images/missingbar_with_targets.png")
