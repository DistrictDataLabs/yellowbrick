import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.features import JointPlotVisualizer


def jointplot(X, y, outpath, **kwargs):
    # Create the visualizer
    visualizer = JointPlotVisualizer(**kwargs)
    visualizer.fit(X, y)
    visualizer.transform(X)

    # Save to disk
    visualizer.poof(outpath=outpath)
    plt.savefig(outpath)


if __name__ == '__main__':

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
