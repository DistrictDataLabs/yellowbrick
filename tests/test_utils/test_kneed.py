# tests.test_utils.test_kneed
# A port of the tests for knee-point detection package, kneed.
#
# Author:   Kevin Arvai
# Author:   Pradeep Singh
# Created:  Mon Apr 23 01:29:18 2019 -0400
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
# ID: test_kneed.py [] pswaldia@no-reply.github.com $

"""
This package contains a port of the tests for knee-point detection package, kneed, by
Kevin Arvai and hosted at https://github.com/arvkevi/kneed. This port is maintained
with permission by the Yellowbrick contributors.
"""

import numpy as np
from yellowbrick.utils.kneed import KneeLocator

x = np.arange(0, 10)
y_convex_inc = np.array([1, 2, 3, 4, 5, 10, 15, 20, 40, 100])
y_convex_dec = np.array(y_convex_inc[::-1])
y_concave_dec = np.array(100 - y_convex_inc)
y_concave_inc = np.array(100 - y_convex_dec)


def test_concave_increasing():
    """Tests that a correct knee point is detected in
    curve having concave and increasing nature."""
    kn = KneeLocator(
        x, y_concave_inc, curve_nature="concave", curve_direction="increasing"
    )
    assert kn.knee == 2


def test_concave_decreasing():
    """Tests that a correct knee point is detected in
    curve having concave and decreasing nature."""
    kn = KneeLocator(
        x, y_concave_dec, curve_nature="concave", curve_direction="decreasing"
    )
    assert kn.knee == 7


def test_convex_increasing():
    """Tests that a correct knee point is detected in
    curve having convex and increasing nature."""
    kn = KneeLocator(
        x, y_convex_inc, curve_nature="convex", curve_direction="increasing"
    )
    assert kn.knee == 7


def test_convex_decreasing():
    """Tests that a correct knee point is detected in
    curve having convex and decreasing nature."""
    kn = KneeLocator(
        x, y_convex_dec, curve_nature="convex", curve_direction="decreasing"
    )
    assert kn.knee == 2


def test_concave_increasing_truncated():
    """Tests that a correct knee point is detected in
    curve having truncated concave increasing nature"""
    kn = KneeLocator(
        x[:-3] / 10,
        y_concave_inc[:-3] / 10,
        curve_nature="concave",
        curve_direction="increasing",
    )
    assert kn.knee == 0.2


def test_concave_decreasing_truncated():
    """Tests that a correct knee point is detected in
    curve having truncated concave decreasing nature"""
    kn = KneeLocator(
        x[:-3] / 10,
        y_concave_dec[:-3] / 10,
        curve_nature="concave",
        curve_direction="decreasing",
    )
    assert kn.knee == 0.4


def test_convex_increasing_truncated():
    """Tests that a correct knee point is detected in
    curve having truncated convex increasing nature"""
    kn = KneeLocator(
        x[:-3] / 10,
        y_convex_inc[:-3] / 10,
        curve_nature="convex",
        curve_direction="increasing",
    )
    assert kn.knee == 0.4


def test_convex_decreasing_truncated():
    """Tests that a correct knee point is detected in
    curve having truncated convex decreasing nature"""
    kn = KneeLocator(
        x[:-3] / 10,
        y_convex_dec[:-3] / 10,
        curve_nature="convex",
        curve_direction="decreasing",
    )
    assert kn.knee == 0.2
