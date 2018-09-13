# yellowbrick.target.binning
# Generates images for the balance binning reference documentation.
#
# Author:  Kristen McIntyre <kautumn06@gmail.com>
# Created: Tue Sept 11 12:09:40 2018 -0400
#
# ID: binning.py [] kautumn06@gmail.com $

"""
Generates images for the balanced binning reference documentation.
"""

##########################################################################
## Imports
##########################################################################

from yellowbrick.target import BalancedBinningReference 
from sklearn.datasets import load_diabetes


def balanced_binning_reference(path="images/balanced_binning_reference.png"):
    # Load a regression data set
    data = load_diabetes()

    # Extract the target variable
    y = data['target']

    # Instantiate and fit the visualizer
    visualizer = BalancedBinningReference()
    visualizer.fit(y)
    return visualizer.poof(outpath=path)



if __name__ == '__main__':
    balanced_binning_reference()
