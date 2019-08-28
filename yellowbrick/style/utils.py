# yellowbrick.style.utils
# Utility functions for styles
#
# Author:   Neal Humphrey
# Created:  Wed Mar 22 12:39:35 2017 -0400
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: utils.py [45268fc] humphrey.neal@gmail.com $

"""
Utility functions for styles
"""

##########################################################################
## Imports
##########################################################################

import numpy as np


def find_text_color(base_color, dark_color="black", light_color="white", coef_choice=0):
    """
    Takes a background color and returns the appropriate light or dark text color.
    Users can specify the dark and light text color, or accept the defaults of 'black' and 'white'

    base_color: The color of the background. This must be
        specified in RGBA with values between 0 and 1 (note, this is the default
        return value format of a call to base_color = cmap(number) to get the
        color corresponding to a desired number). Note, the value of `A` in RGBA
        is not considered in determining light/dark.

    dark_color: Any valid matplotlib color value.
        Function will return this value if the text should be colored dark

    light_color: Any valid matplotlib color value.
        Function will return this value if thet text should be colored light.

    coef_choice: slightly different approaches to calculating brightness. Currently two options in
        a list, user can enter 0 or 1 as list index. 0 is default.
    """

    # Coefficients:
    # option 0: http://www.nbdtech.com/Blog/archive/2008/04/27/Calculating-the-Perceived-Brightness-of-a-Color.aspx
    # option 1: http://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
    coef_options = [
        np.array((0.241, 0.691, 0.068, 0)),
        np.array((0.299, 0.587, 0.114, 0)),
    ]

    coefs = coef_options[coef_choice]
    rgb = np.array(base_color) * 255
    brightness = np.sqrt(np.dot(coefs, rgb ** 2))

    # Threshold from option 0 link; determined by trial and error.
    # base is light
    if brightness > 130:
        return dark_color
    return light_color
