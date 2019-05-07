import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.features import SingleFeatureViz

def sfvplot(X, y, outpath, **kwargs):
    # Create the visualizer
    visualizer = SingleFeatureViz(**kwargs)
    visualizer.fit(X, y)

    # Save to disk
    visualizer.poof(outpath=outpath)
    plt.savefig(outpath)
    plt.clf()

if __name__ == '__main__':

    # Load the regression data set
    data = pd.read_csv("../../../examples/data/occupancy/occupancy.csv")

    features = ["temperature", "relative humidity", "light", "C02", "humidity"]

    # Draw the joint plot visualizer
    sfvplot(data, None, "images/singlefeatureviz_violin.png", idx="temperature", features=features, plot_type="violin")
    sfvplot(data, None, "images/singlefeatureviz_hist.png", idx="temperature", features=features, plot_type="hist")
    sfvplot(data, None, "images/singlefeatureviz_box.png", idx="temperature", features=features, plot_type="box")

