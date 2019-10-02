# tests.test_utils.test_timer
# Tests for the stand alone timer functions in Yellowbrick utils.
#
# Author:   ZJ Poh <poh.zijie@gmail.com>
# Created:  Tue Jul 17 21:11:11 2018 -0700
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
"""
Tests for the stand alone timer functions in Yellowbrick utils.
"""

##########################################################################
## Imports
##########################################################################

import pytest

from unittest import mock
from yellowbrick.utils.timer import *


##########################################################################
## Helper Function Tests
##########################################################################


class TestTimer(object):
    """
    Timer functions and utilities
    """

    @mock.patch("time.time", mock.Mock(side_effect=[1234.2, 1242.8]))
    def test_timer(self):
        with Timer() as timer:
            pass
        assert isinstance(timer.interval, float)
        assert timer.interval == pytest.approx(8.6)


@pytest.mark.parametrize(
    "s,expected",
    [
        (1.01, "00:00:01.0100"),
        (61.01, "00:01:01.0100"),
        (3661.01, "01:01:01.0100"),
        (360061.01, "100:01:01.0100"),
    ],
)
def test_human_readable_time(s, expected):
    assert human_readable_time(s) == expected
