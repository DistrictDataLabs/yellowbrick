
from yellowbrick.classifier.confusion_matrix import *
from tests.base import VisualTestCase


from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression




class ConfusionMatrixTests(VisualTestCase):
    def __init__(self, *args, **kwargs):
        super(ConfusionMatrixTests, self).__init__(*args, **kwargs)
        #Use the same data for all the tests
        self.digits = load_digits()

        X = self.digits.data
        y = self.digits.target
        
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state=11)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def test_confusion_matrix(self):
        model = LogisticRegression()
        cm = ConfusionMatrix(model, classes=[0,1,2,3,4,5,6,7,8,9])
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)

    def test_no_classes_provided(self):
        model = LogisticRegression()
        cm = ConfusionMatrix(model)
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)

    def test_raw_count_mode(self):
        model = LogisticRegression()
        cm = ConfusionMatrix(model)
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test, percent=False)

    def test_zoomed_in(self):
        model = LogisticRegression()
        cm = ConfusionMatrix(model, classes=[0,1,2])
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)

    def test_extra_classes(self):
        model = LogisticRegression()
        cm = ConfusionMatrix(model, classes=[0,1,2,11])
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)
        self.assertTrue(cm.selected_class_counts[3]==0)

    def test_one_class(self):
        model = LogisticRegression()
        cm = ConfusionMatrix(model, classes=[0])
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)