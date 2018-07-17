
from yellowbrick.classifier.class_balance import *

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin

from sklearn.svm import LinearSVC

##########################################################################
## Data
##########################################################################

X = np.array(
        [[ 2.318, 2.727, 4.260, 7.212, 4.792],
         [ 2.315, 2.726, 4.295, 7.140, 4.783,],
         [ 2.315, 2.724, 4.260, 7.135, 4.779,],
         [ 2.110, 3.609, 4.330, 7.985, 5.595,],
         [ 2.110, 3.626, 4.330, 8.203, 5.621,],
         [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
    )

y = np.array([1, 1, 0, 1, 0, 0])

##########################################################################
##  Tests
##########################################################################

class ClassBalanceTests(VisualTestCase, DatasetMixin):

    def test_class_report(self):
        """
        Assert no errors occur during classification report integration
        """
        model = LinearSVC()
        model.fit(X,y)
        visualizer = ClassBalance(model, classes=["A", "B"])
        visualizer.score(X,y)
        self.assert_images_similar(visualizer)

    def test_score_returns_score(self):
        """
        Test that ClassBalance score() returns a score between 0 and 1
        """
        data = self.load_data("occupancy")
        X = data[[
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]]

        y = data['occupancy']

        # Convert X to an ndarray
        X = X.copy().view((float, len(X.dtype.names)))

        # Create and fit the visualizer
        visualizer = ClassBalance(LinearSVC())
        visualizer.fit(X, y)

        # Score the visualizer
        s = visualizer.score(X, y)

        assert 0 <= s <= 1
