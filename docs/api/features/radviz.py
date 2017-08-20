import pandas as pd
from yellowbrick.features import RadViz


# Load the classification data set
data = pd.read_csv("../../../examples/data/occupancy/occupancy.csv")

# Specify the features of interest and the classes of the target
features = ["temperature", "relative humidity", "light", "C02", "humidity"]
classes = ['unoccupied', 'occupied']

# Extract the numpy arrays from the data frame
X = data[features].as_matrix()
y = data.occupancy.as_matrix()

# Instantiate the visualizer
visualizer = RadViz(classes=classes, features=features)

visualizer.fit(X, y)
visualizer.transform(X)
visualizer.poof(outpath="images/radviz.png")
