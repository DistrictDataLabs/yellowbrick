# yellowbrick.bestfit
# Uses Scikit-Learn to compute a best fit function, then draws it in the plot.
#
# Author:   Benjamin Bengfort
# Created:  Sun Jun 26 17:27:08 2016 -0400
#
# Copyright (C) 2016 The sckit-yb developers
# For license information, see LICENSE.txt
#
# ID: bestfit.py [56236f3] benjamin@bengfort.com $

"""
Uses Scikit-Learn to compute a best fit function, then draws it in the plot.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error as mse

from operator import itemgetter
from yellowbrick.style.palettes import LINE_COLOR
from yellowbrick.exceptions import YellowbrickValueError


##########################################################################
## Module Constants
##########################################################################

# Names of the various estimator functions
LINEAR = "linear"
QUADRATIC = "quadratic"
EXPONENTIAL = "exponential"
LOG = "log"
SELECT_BEST = "select_best"


##########################################################################
## Draw Line of Best Fit
##########################################################################


def draw_best_fit(X, y, ax, estimator="linear", **kwargs):
    """
    Uses Scikit-Learn to fit a model to X and y then uses the resulting model
    to predict the curve based on the X values. This curve is drawn to the ax
    (matplotlib axis) which must be passed as the third variable.

    The estimator function can be one of the following:

    - ``'linear'``:      Uses OLS to fit the regression
    - ``'quadratic'``:   Uses OLS with Polynomial order 2
    - ``'exponential'``: Not implemented yet
    - ``'log'``:         Not implemented yet
    - ``'select_best'``: Selects the best fit via MSE

    The remaining keyword arguments are passed to ax.plot to define and
    describe the line of best fit.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    estimator : string, default: 'linear'
        The name of the estimator function used to draw the best fit line.
        The estimator can currently be one of linear, quadratic, exponential,
        log, or select_best. The select best method uses the minimum MSE to
        select the best fit line.

    kwargs : dict
        Keyword arguments to pass to the matplotlib plot function to style and
        label the line of best fit. By default, the standard line color is
        used unless the color keyword argument is passed in.

    Returns
    -------

    ax : matplotlib Axes
        The axes with the line drawn on it.
    """

    # Estimators are the types of best fit lines that can be drawn.
    estimators = {
        LINEAR: fit_linear,  # Uses OLS to fit the regression
        QUADRATIC: fit_quadratic,  # Uses OLS with Polynomial order 2
        EXPONENTIAL: fit_exponential,  # Not implemented yet
        LOG: fit_log,  # Not implemented yet
        SELECT_BEST: fit_select_best,  # Selects the best fit via MSE
    }

    # Check to make sure that a correct estimator value was passed in.
    if estimator not in estimators:
        raise YellowbrickValueError(
            "'{}' not a valid type of estimator; choose from {}".format(
                estimator, ", ".join(estimators.keys())
            )
        )

    # Then collect the estimator function from the mapping.
    estimator = estimators[estimator]

    # Ensure that X and y are the same length
    if len(X) != len(y):
        raise YellowbrickValueError(
            (
                "X and y must have same length:" " X len {} doesn't match y len {}!"
            ).format(len(X), len(y))
        )

    # Ensure that X and y are np.arrays
    X = np.array(X)
    y = np.array(y)

    # Verify that X is a two dimensional array for Scikit-Learn esitmators
    # and that its dimensions are (n, 1) where n is the number of rows.
    if X.ndim < 2:
        X = X[:, np.newaxis]  # Reshape X into the correct dimensions

    if X.ndim > 2:
        raise YellowbrickValueError(
            "X must be a (1,) or (n,1) dimensional array not {}".format(X.shape)
        )

    # Verify that y is a (n,) dimensional array
    if y.ndim > 1:
        raise YellowbrickValueError(
            "y must be a (1,) dimensional array not {}".format(y.shape)
        )

    # Uses the estimator to fit the data and get the model back.
    model = estimator(X, y)

    # Set the color if not passed in.
    if "c" not in kwargs and "color" not in kwargs:
        kwargs["color"] = LINE_COLOR

    # Get the current working axes
    ax = ax or plt.gca()

    # Plot line of best fit onto the axes that were passed in.
    # TODO: determine if xlim or X.min(), X.max() are better params
    xr = np.linspace(*ax.get_xlim(), num=100)
    ax.plot(xr, model.predict(xr[:, np.newaxis]), **kwargs)
    return ax


##########################################################################
## Estimator Functions
##########################################################################


def fit_select_best(X, y):
    """
    Selects the best fit of the estimators already implemented by choosing the
    model with the smallest mean square error metric for the trained values.
    """
    models = [fit(X, y) for fit in [fit_linear, fit_quadratic]]
    errors = map(lambda model: mse(y, model.predict(X)), models)

    return min(zip(models, errors), key=itemgetter(1))[0]


def fit_linear(X, y):
    """
    Uses OLS to fit the regression.
    """
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model


def fit_quadratic(X, y):
    """
    Uses OLS with Polynomial order 2.
    """
    model = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
    model.fit(X, y)
    return model


def fit_exponential(X, y):
    """
    Fits an exponential curve to the data.
    """
    raise NotImplementedError("Exponential best fit lines are not implemented")


def fit_log(X, y):
    """
    Fit a logrithmic curve to the data.
    """
    raise NotImplementedError("Logrithmic best fit lines are not implemented")


##########################################################################
## Draw 45 Degree Line
##########################################################################


def draw_identity_line(ax=None, dynamic=True, **kwargs):
    """
    Draws a 45 degree identity line such that y=x for all points within the
    given axes x and y limits. This function also registeres a callback so
    that as the figure is modified, the axes are updated and the line remains
    drawn correctly.

    Parameters
    ----------

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    dynamic : bool, default : True
        If the plot is dynamic, callbacks will be registered to update the
        identiy line as axes are changed.

    kwargs : dict
        Keyword arguments to pass to the matplotlib plot function to style the
        identity line.


    Returns
    -------

    ax : matplotlib Axes
        The axes with the line drawn on it.

    Notes
    -----

    .. seealso:: `StackOverflow discussion: Does matplotlib have a function for drawing diagonal lines in axis coordinates? <https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates>`_
    """

    # Get the current working axes
    ax = ax or plt.gca()

    # Define the standard line color
    if "c" not in kwargs and "color" not in kwargs:
        kwargs["color"] = LINE_COLOR

    # Define the standard opacity
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.5

    # Draw the identity line
    identity, = ax.plot([], [], **kwargs)

    # Define the callback
    def callback(ax):
        # Get the x and y limits on the axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Set the bounding range of the line
        data = (max(xlim[0], ylim[0]), min(xlim[1], ylim[1]))
        identity.set_data(data, data)

    # Register the callback and return
    callback(ax)

    if dynamic:
        ax.callbacks.connect("xlim_changed", callback)
        ax.callbacks.connect("ylim_changed", callback)

    return ax


if __name__ == "__main__":
    import os
    import pandas as pd

    path = os.path.join(
        os.path.dirname(__file__), "..", "examples", "data", "concrete.xls"
    )
    if not os.path.exists(path):
        raise Exception("Could not find path for testing")

    xkey = "Fine Aggregate (component 7)(kg in a m^3 mixture)"
    ykey = "Coarse Aggregate  (component 6)(kg in a m^3 mixture)"
    data = pd.read_excel(path)

    fig, axe = plt.subplots()
    axe.scatter(data[xkey], data[ykey])
    draw_best_fit(data[xkey], data[ykey], axe, "select_best")

    plt.show()
