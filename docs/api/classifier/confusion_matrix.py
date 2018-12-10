from sklearn.datasets import load_digits, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts

from yellowbrick.classifier import ConfusionMatrix


if __name__ == '__main__':
    digits = load_digits()
    digit_X = digits.data
    digit_y = digits.target
    d_X_train, d_X_test, d_y_train, d_y_test = tts(
        digit_X, digit_y, test_size=0.2
    )
    model = LogisticRegression()
    digit_cm = ConfusionMatrix(model, classes=[0,1,2,3,4,5,6,7,8,9])
    digit_cm.fit(d_X_train, d_y_train)
    digit_cm.score(d_X_test, d_y_test)
    d = digit_cm.poof(outpath="images/confusion_matrix_digits.png")


    iris = load_iris()
    iris_X = iris.data
    iris_y = iris.target
    iris_classes = iris.target_names
    i_X_train, i_X_test, i_y_train, i_y_test = tts(
        iris_X, iris_y, test_size=0.2
    )
    model = LogisticRegression()
    iris_cm = ConfusionMatrix(
        model, classes=iris_classes,
        label_encoder={0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    )
    iris_cm.fit(i_X_train, i_y_train)
    iris_cm.score(i_X_test, i_y_test)
    i = iris_cm.poof(outpath="images/confusion_matrix_iris.png")
