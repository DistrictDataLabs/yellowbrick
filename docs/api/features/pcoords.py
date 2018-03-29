import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.features import ParallelCoordinates


def pcoords(X, y, outpath, **kwargs):
    # Create a new figure and axes
    _, ax = plt.subplots()

    # Create the visualizer
    visualizer = ParallelCoordinates(ax=ax, **kwargs)
    visualizer.fit(X, y)
    visualizer.transform(X)

    # Save to disk
    visualizer.poof(outpath=outpath)


if __name__ == '__main__':
    # Load the classification data set
    data = pd.read_csv("../../../examples/data/occupancy/occupancy.csv")

    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']

    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.occupancy.as_matrix()

    # Draw the full, original parallel coordinates
    pcoords(X, y, "images/parallel_coordinates.png", classes=classes, features=features)

    # Draw the noramlized, sampled parallel coordinates
    pcoords(X, y, "images/normalized_sampled_parallel_coordinates.png",
        classes=classes, features=features,
        normalize='standard', sample=0.1,
    )
