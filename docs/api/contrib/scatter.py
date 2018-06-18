import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.contrib.scatter import ScatterVisualizer


def scatter(data, target, outpath, **kwargs):
    # Create a new figure and axes
    _, ax = plt.subplots()

    # Create the visualizer
    visualizer = ScatterVisualizer(ax=ax, **kwargs)
    visualizer.fit(data, target)
    visualizer.transform(data)

    # Save to disk
    visualizer.poof(outpath=outpath)
    print(outpath)


if __name__ == '__main__':
    # Load the classification data set
    data = pd.read_csv("../../../examples/data/occupancy/occupancy.csv")

    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']

    # Extract the numpy arrays from the data frame
    X = data[features]
    y = data.occupancy

    # Draw the scatter visualizer
    scatter(X, y, "images/scatter.png", x='light', y='C02', classes=classes)
