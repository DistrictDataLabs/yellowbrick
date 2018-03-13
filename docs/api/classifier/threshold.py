import pandas as pd

from yellowbrick.classifier import ThreshViz
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    # Load the data set
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', header=None)
    data.rename(columns={57:'is_spam'}, inplace=True)

    features = [col for col in data.columns if col != 'is_spam']

    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.is_spam.as_matrix()

    # Instantiate the classification model and visualizer
    logistic = LogisticRegression()
    visualizer = ThreshViz(logistic)

    visualizer.fit(X, y)  # Fit the training data to the visualizer
    g = visualizer.poof(outpath="images/thresholdviz.png") # Draw/show/poof the data
