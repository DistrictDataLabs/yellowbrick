.. -*- mode: rst -*-

Cook's Distance
===============

Cook’s Distance is a measure of an observation or instances’ influence on a
linear regression. Instances with a large influence may be outliers, and
datasets with a large number of highly influential points might not be
suitable for linear regression without further processing such as outlier
removal or imputation. The ``CooksDistance`` visualizer shows a stem plot of
all instances by index and their associated distance score, along with a
heuristic threshold to quickly show what percent of the dataset may be
impacting OLS regression models.

=================   ==============================
Visualizer           :class:`~yellowbrick.regressor.influence.CooksDistance`
Quick Method         :func:`~yellowbrick.regressor.influence.cooks_distance`
Models               General Linear Models
Workflow             Dataset/Sensitivity Analysis
=================   ==============================

.. plot::
    :context: close-figs
    :alt: Cook's distance using concrete dataset

    from yellowbrick.regressor import CooksDistance
    from yellowbrick.datasets import load_concrete

    # Load the regression dataset
    X, y = load_concrete()

    # Instantiate and fit the visualizer
    visualizer = CooksDistance()
    visualizer.fit(X, y)
    visualizer.show()

The presence of so many highly influential points suggests that linear
regression may not be suitable for this dataset. One or more of the four
assumptions behind linear regression might be being violated; namely one of:
independence of observations, linearity of response, normality of residuals,
or homogeneity of variance ("homoscedasticity"). We can check the latter three
conditions using a residual plot:

.. plot::
    :context: close-figs
    :alt: Residual plot using concrete dataset

    from sklearn.linear_model import LinearRegression
    from yellowbrick.regressor import ResidualsPlot

    # Instantiate and fit the visualizer
    model = LinearRegression()
    visualizer_residuals = ResidualsPlot(model)
    visualizer_residuals.fit(X, y)
    visualizer_residuals.show()

The residuals appear to be normally distributed around 0, satisfying the
linearity and normality conditions. However, they do skew slightly positive
for larger predicted values, and also appear to increase in magnitude as the
predicted value increases, suggesting a violation of the homoscedasticity
condition.

Given this information, we might consider one of the following options: (1)
using a linear regression anyway, (2) using a linear regression after removing
outliers, and (3) resorting to other regression models. For the sake of
illustration, we will go with option (2) with the help of the Visualizer’s
public learned parameters ``distance_`` and ``influence_threshold_``:

.. plot::
    :context: close-figs
    :alt: Residual plot using concrete dataset after outlier removal

    i_less_influential = (visualizer.distance_ <= visualizer.influence_threshold_)
    X_li, y_li = X[i_less_influential], y[i_less_influential]

    model = LinearRegression()
    visualizer_residuals = ResidualsPlot(model)
    visualizer_residuals.fit(X_li, y_li)
    visualizer_residuals.show()

The violations of the linear regression assumptions addressed earlier appear
to be diminished. The goodness-of-fit measure has increased from 0.615 to
0.748, which is to be expected as there is less variance in the response
variable after outlier removal.

Quick Method
------------

Similar functionality as above can be achieved in one line using the
associated quick method, ``cooks_distance``. This method will instantiate and
fit a ``CooksDistance`` visualizer on the training data, then will score it on
the optionally provided test data (or the training data if it is not
provided).

.. plot::
    :context: close-figs
    :alt: cooks_distance quick method

    from yellowbrick.datasets import load_concrete
    from yellowbrick.regressor import cooks_distance

    # Load the regression dataset
    X, y = load_concrete()

    # Instantiate and fit the visualizer
    cooks_distance(
        X, y,
        draw_threshold=True,
        linefmt="C0-", markerfmt=","
    )


API Reference
-------------

.. automodule:: yellowbrick.regressor.influence
    :members: CooksDistance, cooks_distance
    :undoc-members:
    :show-inheritance:
