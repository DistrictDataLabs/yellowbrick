import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.features import ParallelCoordinates

if __name__ == '__main__':
    # Load the classification data set
    data = pd.read_csv("../../../examples/data/occupancy/occupancy.csv")

    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']

    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.occupancy.as_matrix()

    # Instantiate the visualizer
    visualizer = ParallelCoordinates(classes=classes, features=features)

    visualizer.fit(X, y)      # Fit the data to the visualizer
    visualizer.transform(X)   # Transform the data
    visualizer.poof(outpath="images/parallel_coordinates.png")         # Draw/show/poof the data
