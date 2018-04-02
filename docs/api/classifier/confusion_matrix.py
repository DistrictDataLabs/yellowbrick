from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from yellowbrick.classifier import ConfusionMatrix


if __name__ == '__main__':
    # Load the regression data set
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state=11)

    model = LogisticRegression()

    #The ConfusionMatrix visualizer taxes a model
    cm = ConfusionMatrix(model, classes=[0,1,2,3,4,5,6,7,8,9])

    cm.fit(X_train, y_train)  # Fit the training data to the visualizer
    cm.score(X_test, y_test)  # Evaluate the model on the test data
    g = cm.poof(outpath="images/confusion_matrix.png")             # Draw/show/poof the data
