# tests.random
# A visualizer that draws a random scatter plot for testing.
#
# Author:  Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created: Wed Mar 21 17:51:15 2018 -0400
#
# ID: rand.py [cc69b3c] davidwaterman@gmail.com $

"""
A visualizer that draws a random scatter plot for testing.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from yellowbrick.base import Visualizer
from yellowbrick.style import resolve_colors

from sklearn.datasets import make_blobs


##########################################################################
## Random Visualizer
##########################################################################


class RandomVisualizer(Visualizer):
    """
    Creates random scatter plots as a testing utility.

    Data generation uses scikit-learn make_blobs to create scatter plots that
    have reasonable visual features and multiple colors.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    n_samples : int, default: 100
        The number of points to generate for the scatter plot

    n_blobs : int or array of shape [n_centers, 2]
        Define the number of blobs to create or specify their centers.

    random_state : int, RandomState or None:
        Used to specify the seed of the random state to ensure tests work.
    """

    def __init__(self, ax=None, n_samples=100, n_blobs=3, random_state=None, **kwargs):

        super(RandomVisualizer, self).__init__(ax=ax, **kwargs)
        if isinstance(random_state, (int, float)) or random_state is None:
            random_state = np.random.RandomState(random_state)

        self.n_samples = n_samples
        self.n_blobs = n_blobs
        self.random_state = random_state

    def generate(self):
        """
        Returns random data according to the visualizer specification.

        Returns
        -------
        X : array of shape [n_samples, 2]
            2 dimensional array of points to plot

        y : vector with length n_samples
            Center/blob each point belongs to (used for color)
        """
        return make_blobs(
            self.n_samples, 2, self.n_blobs, random_state=self.random_state
        )

    def fit(self, *args, **kwargs):
        X, c = self.generate()

        x = X[:, 0]
        y = X[:, 1]

        self.draw(x, y, c)
        return self

    def draw(self, x, y, c):
        colors = resolve_colors(self.n_blobs)

        for i in np.arange(self.n_blobs):
            mask = c == i
            label = "c{}".format(i)
            self.ax.scatter(x[mask], y[mask], label=label, c=colors[i])

        return self.ax

    def finalize(self):
        self.ax.legend(frameon=True)
        self.ax.set_ylabel("$y$")
        self.ax.set_xlabel("$x$")
        self.ax.set_title("Random Scatter Plot")
        return self.ax


if __name__ == "__main__":
    r = RandomVisualizer()
    r.fit()
    r.show(outpath="test.png")
