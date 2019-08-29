# yellowbrick.utils.timer
# Timer utilities
#
# Author:   ZJ Poh
# Created:  Mon Jul 16 10:51:13 2017 -0700
#
# Copyright (C) 2017 The sckit-yb developers
# For license information, see LICENSE.txt
#
# ID: timer.py [75b0f6a] 8103276+zjpoh@users.noreply.github.com $
"""
Timer utilities
"""

##########################################################################
## Imports
##########################################################################

import time

##########################################################################
## Timer Class
##########################################################################


def human_readable_time(s):
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return "{:>02.0f}:{:02.0f}:{:>07.4f}".format(h, m, s)


class Timer:
    """
    A context object timer. Usage:
        >>> with Timer() as timer:
        ...     do_something()
        >>> print(timer.interval)
    """

    def __init__(self):
        self.time = time.time

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, *exc):
        self.finish = self.time()
        self.interval = self.finish - self.start

    def __str__(self):
        return human_readable_time(self.interval)
