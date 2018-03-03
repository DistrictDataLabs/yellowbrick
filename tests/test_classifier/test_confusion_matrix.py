import yellowbrick
import matplotlib.pyplot as plt

from yellowbrick.classifier.confusion_matrix import *

from tests.base import VisualTestCase

from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.model_selection import train_test_split as tts


class ConfusionMatrixTests(VisualTestCase):
    """
    ConfusionMatrix visualizer
    """

    def setUp(self):
        #Use the same data for all the tests
        self.digits = load_digits()

        X = self.digits.data
        y = self.digits.target

        X_train, X_test, y_train, y_test = tts(X,y, test_size =0.2, random_state=11)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def test_confusion_matrix(self):
        """
        Integration test of visualizer
        """
        fig = plt.figure()
        ax = fig.add_subplot()

        model = LogisticRegression()
        cm = ConfusionMatrix(model, ax=ax, classes=[0,1,2,3,4,5,6,7,8,9])
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)

        self.assert_images_similar(cm)

    def test_no_classes_provided(self):
        """
        Assert no errors when no classes are provided
        """
        model = LogisticRegression()
        cm = ConfusionMatrix(model)
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)

    def test_raw_count_mode(self):
        """
        Assert that raw count mode works as expected
        """
        model = LogisticRegression()
        cm = ConfusionMatrix(model)
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test, percent=False)

    def test_zoomed_in(self):
        """
        Test zoomed in classes works as expected
        """
        model = LogisticRegression()
        cm = ConfusionMatrix(model, classes=[0,1,2])
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)

    def test_extra_classes(self):
        """
        Test that extra classes are ignored
        """
        model = LogisticRegression()
        cm = ConfusionMatrix(model, classes=[0,1,2,11])
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)
        self.assertTrue(cm.selected_class_counts[3]==0)

    def test_one_class(self):
        """
        Test single class confusion matrix
        """
        fig = plt.figure()
        ax = fig.add_subplot()

        model = LogisticRegression()
        cm = ConfusionMatrix(model, ax=ax, classes=[0])
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)

        self.assert_images_similar(cm)

    def test_defined_mapping(self):
        """
        Test mapping as label encoder
        """
        model = LogisticRegression()
        classes = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        mapping = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                   6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}
        cm = ConfusionMatrix(model, classes=classes, label_encoder = mapping)
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)

    def test_inverse_mapping(self):
        """
        Test LabelEncoder as label encoder
        """
        fig = plt.figure()
        ax = fig.add_subplot()

        model = LogisticRegression()
        le = LabelEncoder()
        classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        le.fit(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])

        cm = ConfusionMatrix(model, ax=ax, classes=classes, label_encoder=le)
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)

        self.assert_images_similar(cm)


    def test_isclassifier(self):
        """
        Test taht non-classifiers raise exceptions
        """
        model = PassiveAggressiveRegressor()
        message = 'This estimator is not a classifier; try a regression or clustering score visualizer instead!'
        classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

        with self.assertRaisesRegexp(yellowbrick.exceptions.YellowbrickError, message):
            ConfusionMatrix(model, classes=classes)
