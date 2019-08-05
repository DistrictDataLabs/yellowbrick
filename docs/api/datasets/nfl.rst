.. -*- mode: rst -*-

NFL
===

This dataset is comprised of statistics on all eligible receivers from the 2018 NFL regular season.

=================   ==================
Samples total                      494
Dimensionality                      20
Features                      str, int
Targets                            N/A
Task(s)                     clustering
=================   ==================

Description
-----------

The dataset consists of an aggregate of all relevant statistics for eligible receivers that played in at least 1 game and had at least 1 target throughout the season. This is not limited to players specifically designated as wide-receivers, but may include other positions such as running-backs and tight-ends.

Citation
--------

Redistributed with the permission of Sports Reference LLC on June 11, 2019 via email.

Sports Reference LLC, "2018 NFL Receiving," Pro-Football-Reference.com - Pro Football Statistics and History.
[Online]. Available `here <https://www.pro-football-reference.com/years/2018/receiving.htm>`_. [Accessed: 18-Jun-2019]

Loader
------

.. autofunction:: yellowbrick.datasets.loaders.load_nfl
