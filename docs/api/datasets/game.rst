.. -*- mode: rst -*-

Game
====

The dataset was created and donated to the UCI ML Repository by John Tromp (tromp '@' cwi.nl).

=================   ==============================
Samples total                                67557
Dimensionality                                  42
Features                               categorical
Targets               str: {"win", "loss", "draw"}
Task(s)                             classification
=================   ==============================

Description
-----------

This database contains all legal 8-ply positions in the game of connect-4 in which neither player has won yet, and in which the next move is not forced.

The symbol x represents the first player; o the second. The dataset contains the state of the game by representing each position in a 6x7 grid board. The outcome class is the game theoretical value for the first player.

Citation
--------

Downloaded from the `UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/datasets/Connect-4>`_  on May 4, 2017.

Loader
------

.. autofunction:: yellowbrick.datasets.loaders.load_game
