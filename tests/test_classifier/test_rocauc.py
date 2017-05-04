
from yellowbrick.classifier.rocauc import *
from tests.base import VisualTestCase
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

class ROCAUCTests(VisualTestCase):

    def test_roc_auc(self):
        """
        Assert no errors occur during ROC-AUC integration
        """
        model = LinearSVC()
        model.fit(X,y)
        visualizer = ROCAUC(model, classes=["A", "B"])
        visualizer.score(X,y)
