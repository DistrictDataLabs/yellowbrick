# tests.test_datasets.test_signature
# Test the sha256sum file signature library
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Tue Jul 31 14:20:10 2018 -0400
#
# ID: test_signature.py [7082742] benjamin@bengfort.com $

"""
Test the sha256sum file signature library
"""

##########################################################################
## Imports
##########################################################################

import json
from yellowbrick.datasets.signature import sha256sum


##########################################################################
## Test Case
##########################################################################

FIXTURE = {
    "name": "The Cat in the Hat",
    "color": "red and black",
    "weather": "rainy",
    "chaos_level": "HIGH",
    "things": ["1", "2"],
    "extra": "angry fish in bowl",
}


def test_signature(tmpdir):
    """
    Test the SHA 256 signature of a temporary file
    """

    fpath = tmpdir.join("test.json")
    json.dump(FIXTURE, fpath, indent=2)
    assert (
        sha256sum(str(fpath))
        == "d10b36aa74a59bcf4a88185837f658afaf3646eff2bb16c3928d0e9335e945d2"
    )
