import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.features import ScatterVisualizer, JointPlotVisualizer


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


def jointplot(X, y, outpath, **kwargs):
    # Create a new figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Create the visualizer
    visualizer = JointPlotVisualizer(ax=ax, **kwargs)
    visualizer.fit(X, y)
    visualizer.transform(X)

    # Save to disk
    visualizer.poof(outpath=outpath)
    plt.savefig(outpath)


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

    # Load the regression data set
    data = pd.read_csv("../../../examples/data/concrete/concrete.csv")

    feature = 'cement'
    target = 'strength'

    # Get the X and y data from the DataFrame
    Xs = data[feature]
    ys = data[target]

    # Draw the joint plot visualizer
    jointplot(Xs, ys, "images/jointplot.png", feature=feature, target=target)

    # Draw the joint plot visualizer with hexadecimal scatter plot
    jointplot(Xs, ys, "images/jointplot_hex.png", feature=feature, target=target, joint_plot='hex')
