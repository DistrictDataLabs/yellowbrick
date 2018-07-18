# tests.test_utils.test_timer
# Tests for the stand alone timer functions in Yellowbrick utils.
#
# Author:   ZJ Poh <poh.zijie@gmail.com>
# Created:  Tue Jul 17 21:11:11 2018 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
"""
Tests for the stand alone timer functions in Yellowbrick utils.
"""

##########################################################################
## Imports
##########################################################################

import unittest
from unittest import mock
import time

from yellowbrick.utils.timer import *

##########################################################################
## Helper Function Tests
##########################################################################

class TestTimer(unittest.TestCase):
    """
    Timer functions and utilities
    """

    @mock.patch('time.time', mock.MagicMock(return_value=1234.))
    def test_timer(self):
        with Timer() as timer:
            timer.time = mock.MagicMock(return_value=1236.)
        self.assertIsInstance(timer.interval, float)
        self.assertEqual(timer.interval, 2)


    def test_human_readable_time(self):
        self.assertEqual(human_readable_time(1.01), '00:00:01.0100')
        self.assertEqual(human_readable_time(61.01), '00:01:01.0100')
        self.assertEqual(human_readable_time(3661.01), '01:01:01.0100')
        self.assertEqual(human_readable_time(360061.01), '100:01:01.0100')
