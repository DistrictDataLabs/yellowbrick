#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:00:12 2019

@author: sarthak
"""

from yellowbrick.features.manifold import manifold_embedding
from yellowbrick.datasets import load_occupancy

#Load the classification dataset
X, y = load_occupancy()

# Specify the target classes
classes = ["unoccupied", "occupied"]

# Instantiate the visualizer
manifold_embedding(X, y, classes=classes)
