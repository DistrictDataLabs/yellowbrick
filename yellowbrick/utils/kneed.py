# yellowbrick.utils.kneed
# A port of the knee-point detection package, kneed.
#
# Author:   Kevin Arvai
# Author:   Pradeep Singh 
# Created:  Mon Apr 15 09:43:18 2019 -0400
#
# Copyright (C) 2017 Kevin Arvai
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list
# of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this 
# list of conditions and the following disclaimer in the documentation and/or other 
# materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may 
# be used to endorse or promote products derived from this software without specific
# prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS 
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ID: kneed.py [] pswaldia@no-reply.github.com $

"""
This package contains a port of the knee-point detection package, kneed, by 
Kevin Arvai and hosted at https://github.com/arvkevi/kneed. This port is maintained 
with permission by the Yellowbrick contributors.
"""
import numpy as np
from scipy import interpolate
from scipy.signal import argrelextrema
import warnings

from yellowbrick.exceptions import YellowbrickWarning


class KneeLocator(object):
    """
    Finds the "elbow" or "knee" which is a value corresponding to the point of maximum curvature 
    in an elbow curve, using knee point detection algorithm. This point is accessible via the
    `knee` attribute.

    Parameters
    ----------
    x : list
       A list of k values representing the no. of clusters in KMeans Clustering algorithm.

    y : list
       A list of silhouette score corresponding to each value of k.
    
    S : float, default: 1.0
       Sensitivity parameter that allows us to adjust how aggressive we want KneeLocator to 
       be when detecting "knees" or "elbows".

    curve_nature : string, default: 'convace'
       A string that determines the nature of the elbow curve in which "knee" or "elbow" is 
       to be found.

    curve_direction : string, default: 'increasing'
       A string that determines tha increasing or decreasing nature of the elbow curve in 
       which "knee" or "elbow" is to be found.
    
    Notes
    -----
    The KneeLocator is implemented using the "knee point detection algorithm" which can be read at
    `<https://www1.icsi.berkeley.edu/~barath/papers/kneedle-simplex11.pdf>`

    """

    def __init__(self, x, y, S=1.0, curve_nature='concave', curve_direction='increasing'):

        # Raw Input
        self.x = x
        self.y = y
        self.curve_nature = curve_nature
        self.curve_direction = curve_direction
        self.N = len(self.x)
        self.S = S

        # Step 1: fit a smooth line
        uspline = interpolate.interp1d(self.x, self.y)
        self.Ds_x = np.linspace(np.min(self.x), np.max(self.x), self.N)
        self.Ds_y = uspline(self.Ds_x)

        # Step 2: normalize values
        self.xsn = self.__normalize(self.Ds_x)
        self.ysn = self.__normalize(self.Ds_y)

        # Step 3: Calculate difference curve
        self.xd = self.xsn
        if self.curve_nature == 'convex' and curve_direction == 'decreasing':
            self.yd = self.ysn + self.xsn
            self.yd = 1 - self.yd
        elif self.curve_nature == 'concave' and curve_direction == 'decreasing':
            self.yd = self.ysn + self.xsn
        elif self.curve_nature == 'concave' and curve_direction == 'increasing':
            self.yd = self.ysn - self.xsn
        if self.curve_nature == 'convex' and curve_direction == 'increasing':
            self.yd = abs(self.ysn - self.xsn)

        # Step 4: Identify local maxima/minima
        # local maxima
        self.xmx_idx = argrelextrema(self.yd, np.greater)[0]
        self.xmx = self.xd[self.xmx_idx]
        self.ymx = self.yd[self.xmx_idx]

        # local minima
        self.xmn_idx = argrelextrema(self.yd, np.less)[0]
        self.xmn = self.xd[self.xmn_idx]
        self.ymn = self.yd[self.xmn_idx]

        # Step 5: Calculate thresholds
        self.Tmx = self.__threshold(self.ymx)

        # Step 6: find knee
        self.knee, self.norm_knee, self.knee_x = self.find_knee()

    @staticmethod
    def __normalize(a):
        """
        Normalizes an array.

        Parameters
        -----------
        a : list 
           The array to normalize
        """
        return (a - min(a)) / (max(a) - min(a))

    def __threshold(self, ymx_i):
        """
        Calculates the difference threshold for a
        given difference local maximum.

        Parameters
        -----------
        ymx_i : float 
           The normalized y value of a local maximum.
        """
        return ymx_i - (self.S * np.diff(self.xsn).mean())

    def find_knee(self, ):
        """
        Finds and returns the "knee"or "elbow" value, the normalized knee
        value, and the x value where the knee is located.

        """
        if not self.xmx_idx.size:
            warning_message = \
                'No "knee" or "elbow point" detected ' \
                'This could be due to bad clustering, no '\
                'actual clusters being formed etc.'        
            warnings.warn(warning_message,YellowbrickWarning)
            return None, None, None

        mxmx_iter = np.arange(self.xmx_idx[0], len(self.xsn))
        xmx_idx_iter = np.append(self.xmx_idx, len(self.xsn))

        knee_, norm_knee_, knee_x = 0.0, 0.0, None
        for mxmx_i, mxmx in enumerate(xmx_idx_iter):
            # stopping criteria for exhasuting array
            if mxmx_i == len(xmx_idx_iter) - 1:
                break
            # indices between maxima/minima
            idxs = (mxmx_iter > xmx_idx_iter[mxmx_i]) * \
                (mxmx_iter < xmx_idx_iter[mxmx_i + 1])
            between_local_mx = mxmx_iter[np.where(idxs)]

            for j in between_local_mx:
                if j in self.xmn_idx:
                    # reached a minima, x indices are unique
                    # only need to check if j is a min
                    if self.yd[j + 1] > self.yd[j]:
                        self.Tmx[mxmx_i] = 0
                        knee_x = None  # reset x where yd crossed Tmx
                    elif self.yd[j + 1] <= self.yd[j]:
                        warning_message="If this is a minima, " \
                                        "how would you ever get here."
                        warnings.warn(warning_message, YellowbrickWarning)
                if self.yd[j] < self.Tmx[mxmx_i] or self.Tmx[mxmx_i] < 0:
                    # declare a knee
                    if not knee_x:
                        knee_x = j
                    knee_ = self.x[self.xmx_idx[mxmx_i]]
                    norm_knee_ = self.xsn[self.xmx_idx[mxmx_i]]
        return knee_, norm_knee_, knee_x

    def plot_knee_normalized(self, ):
        """
        Plots the normalized curve, the distance curve (xd, ysn) and the
        knee, if it exists.

        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.plot(self.xsn, self.ysn)
        plt.plot(self.xd, self.yd, 'r')
        plt.xticks(np.arange(min(self.xsn), max(self.xsn) + 0.1, 0.1))
        plt.yticks(np.arange(min(self.xd), max(self.ysn) + 0.1, 0.1))

        plt.vlines(self.norm_knee, plt.ylim()[0], plt.ylim()[1])

    def plot_knee(self, ):
        """
        Plot the curve and the knee, if it exists
        
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.plot(self.x, self.y)
        plt.vlines(self.knee, plt.ylim()[0], plt.ylim()[1])

    # Niceties for users working with elbows rather than knees

    @property
    def elbow(self):
        return self.knee

    @property
    def norm_elbow(self):
        return self.norm_knee

    @property
    def elbow_x(self):
        return self.knee_x


