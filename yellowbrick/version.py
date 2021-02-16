# yellowbrick.version
# Maintains version and package information for deployment.
#
# Author:   Benjamin Bengfort
# Created:  Mon Jan 25 14:22:52 2016 -0500
#
# Copyright (C) 2016 The sckit-yb developers
# For license information, see LICENSE.txt
#
# ID: version.py [0c5ba04] benjamin@bengfort.com $

"""
Maintains version and package information for deployment.
"""

##########################################################################
## Module Info
##########################################################################

__version_info__ = {
    "major": 1,
    "minor": 3,
    "micro": 0,
    "releaselevel": "final",
    "post": 1,
    "serial": 21,
}

##########################################################################
## Helper Functions
##########################################################################


def get_version(short=False):
    """
    Prints the version.
    """
    assert __version_info__["releaselevel"] in ("alpha", "beta", "final")
    vers = ["{major}.{minor}".format(**__version_info__)]

    if __version_info__["micro"]:
        vers.append(".{micro}".format(**__version_info__))

    if __version_info__["releaselevel"] != "final" and not short:
        vers.append(
            "{}{}".format(
                __version_info__["releaselevel"][0],
                __version_info__["serial"],
            )
        )

    if __version_info__["post"]:
        vers.append(".post{}".format(__version_info__["post"]))

    return "".join(vers)
