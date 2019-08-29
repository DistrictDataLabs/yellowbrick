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

    curve_nature : string, default: 'concave'
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

    def __init__(
        self, x, y, S=1.0, curve_nature="concave", curve_direction="increasing"
    ):

        # Raw Input
        self.x = x
        self.y = y
        self.curve_nature = curve_nature
        self.curve_direction = curve_direction
        self.N = len(self.x)
        self.S = S
        self.all_knees = set()
        self.all_norm_knees = set()

        # Step 1: fit a smooth line
        uspline = interpolate.interp1d(self.x, self.y)
        self.x = np.array(x)
        self.Ds_y = uspline(self.x)

        # Step 2: normalize values
        self.x_normalized = self.__normalize(self.x)
        self.y_normalized = self.__normalize(self.Ds_y)

        # Step 3: Calculate the Difference curve
        self.x_normalized, self.y_normalized = self.transform_xy(
            self.x_normalized,
            self.y_normalized,
            self.curve_direction,
            self.curve_nature,
        )
        # normalized difference curve
        self.y_distance = self.y_normalized - self.x_normalized
        self.x_distance = self.x_normalized.copy()

        # Step 4: Identify local maxima/minima
        # local maxima
        self.maxima_inidices = argrelextrema(self.y_distance, np.greater)[0]
        self.x_distance_maxima = self.x_distance[self.maxima_inidices]
        self.y_distance_maxima = self.y_distance[self.maxima_inidices]

        # local minima
        self.minima_indices = argrelextrema(self.y_distance, np.less)[0]
        self.x_distance_minima = self.x_distance[self.minima_indices]
        self.y_distance_minima = self.y_distance[self.minima_indices]

        # Step 5: Calculate thresholds
        self.Tmx = self.y_distance_maxima - (
            self.S * np.abs(np.diff(self.x_normalized).mean())
        )

        # Step 6: find knee
        self.find_knee()
        if (self.all_knees or self.all_norm_knees) == set():
            warning_message = (
                "No 'knee' or 'elbow point' detected "
                "This could be due to bad clustering, no "
                "actual clusters being formed etc."
            )
            warnings.warn(warning_message, YellowbrickWarning)
            self.knee = None
            self.norm_knee = None
        else:
            self.knee, self.norm_knee = min(self.all_knees), min(self.all_norm_knees)

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

    @staticmethod
    def transform_xy(x, y, direction, curve):
        """transform x and y to concave, increasing based on curve_direction and curve_nature"""
        # convert elbows to knees
        if curve == "convex":
            x = x.max() - x
            y = y.max() - y
        # flip decreasing functions to increasing
        if direction == "decreasing":
            y = np.flip(y)

        if curve == "convex":
            x = np.flip(x)
            y = np.flip(y)

        return x, y

    def find_knee(self,):
        """This function finds and sets the knee value and the normalized knee value. """
        if not self.maxima_inidices.size:
            warning_message = (
                'No "knee" or "elbow point" detected '
                "This could be due to bad clustering, no "
                "actual clusters being formed etc."
            )
            warnings.warn(warning_message, YellowbrickWarning)
            return None, None

        # artificially place a local max at the last item in the x_distance array
        self.maxima_inidices = np.append(self.maxima_inidices, len(self.x_distance) - 1)
        self.minima_indices = np.append(self.minima_indices, len(self.x_distance) - 1)

        # placeholder for which threshold region i is located in.
        maxima_threshold_index = 0
        minima_threshold_index = 0
        # traverse the distance curve
        for idx, i in enumerate(self.x_distance):
            # reached the end of the curve
            if i == 1.0:
                break
            # values in distance curve are at or after a local maximum
            if idx >= self.maxima_inidices[maxima_threshold_index]:
                threshold = self.Tmx[maxima_threshold_index]
                threshold_index = idx
                maxima_threshold_index += 1
            # values in distance curve are at or after a local minimum
            if idx >= self.minima_indices[minima_threshold_index]:
                threshold = 0.0
                minima_threshold_index += 1
            # Do not evaluate values in the distance curve before the first local maximum.
            if idx < self.maxima_inidices[0]:
                continue

            # evaluate the threshold
            if self.y_distance[idx] < threshold:
                if self.curve_nature == "convex":
                    if self.curve_direction == "decreasing":
                        knee = self.x[threshold_index]
                        self.all_knees.add(knee)
                        norm_knee = self.x_normalized[threshold_index]
                        self.all_norm_knees.add(norm_knee)
                    else:
                        knee = self.x[-(threshold_index + 1)]
                        self.all_knees.add(knee)
                        norm_knee = self.x_normalized[-(threshold_index + 1)]
                        self.all_norm_knees.add(norm_knee)

                elif self.curve_nature == "concave":
                    if self.curve_direction == "decreasing":
                        knee = self.x[-(threshold_index + 1)]
                        self.all_knees.add(knee)
                        norm_knee = self.x_normalized[-(threshold_index + 1)]
                        self.all_norm_knees.add(norm_knee)
                    else:
                        knee = self.x[threshold_index]
                        self.all_knees.add(knee)
                        norm_knee = self.x_normalized[threshold_index]
                        self.all_norm_knees.add(norm_knee)

    def plot_knee_normalized(self,):
        """
        Plots the normalized curve, the distance curve (x_distance, y_normalized) and the
        knee, if it exists.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.plot(self.x_normalized, self.y_normalized)
        plt.plot(self.x_distance, self.y_distance, "r")
        plt.xticks(
            np.arange(self.x_normalized.min(), self.x_normalized.max() + 0.1, 0.1)
        )
        plt.yticks(np.arange(self.y_distance.min(), self.y_normalized.max() + 0.1, 0.1))

        plt.vlines(self.norm_knee, plt.ylim()[0], plt.ylim()[1])

    def plot_knee(self,):
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
    def all_elbows(self):
        return self.all_knees

    @property
    def all_norm_elbows(self):
        return self.all_norm_knees
