from yellowbrick.classifier import ClassPredictionError

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    # Create classification dataset
    X, y = make_classification(n_samples=1000, n_classes=5,
                               n_informative=3, n_clusters_per_class=1)

    classes = ['apple', 'kiwi', 'pear', 'banana', 'orange']

    # Perform 80/20 training/test split
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.20,
                                           random_state=42)

    # Instantiate the classification model and visualizer
    visualizer = ClassPredictionError(RandomForestClassifier(),
                                      classes=classes)

    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)

    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)

    # Draw visualization
    g = visualizer.poof(outpath="images/class_prediction_error.png")
