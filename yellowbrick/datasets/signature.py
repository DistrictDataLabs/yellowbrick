# yellowbrick.datasets.signature
# Performs SHA 256 hashing of a file for dataset archive verification.
#
# Author:  Benjamin Bengfort
# Created: Tue Jul 31 14:18:11 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: signature.py [7082742] benjamin@bengfort.com $

"""
Performs SHA 256 hashing of a file for dataset archive verification.
"""

##########################################################################
## Imports
##########################################################################

import hashlib

##########################################################################
## Signature checking utility
##########################################################################


def sha256sum(path, blocksize=65536):
    """
    Computes the SHA256 signature of a file to verify that the file has not
    been modified in transit and that it is the correct version of the data.
    """
    sig = hashlib.sha256()
    with open(path, "rb") as f:
        buf = f.read(blocksize)
        while len(buf) > 0:
            sig.update(buf)
            buf = f.read(blocksize)
    return sig.hexdigest()
