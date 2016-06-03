import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

def linearbestfit( xarray=np.array, yarray=np.array):

    # Check xarray and yarray are same length
    if len(xarray) != len(yarray):
        print("Arrays are not the same length")
        exit(1)

    # Verify xarray is (n,1)
    if len(xarray.shape) < 2 :
        xarray=xarray[:,np.newaxis]
    elif len(xarray.shape) > 2:
        print("Incorrect xarray shape. Must be (n,1)")
        exit(1)

    # Verify y array is (n,)
    if len(yarray.shape) != 1:
        print("Inccorect yarray shape. Must be (n,)")
        exit(1)

    # Create, fit, and train regression
    regr = linear_model.LinearRegression()
    regr.fit(xarray, yarray)

    # Plot scatterplot
    plt.scatter(xarray, yarray,  color='black')

    # Plot line of best fit
    plt.plot(xarray, regr.predict(xarray), color='blue',
            linewidth=3)

    # Plot
    plt.xticks(())
    plt.yticks(())
    plt.show()
