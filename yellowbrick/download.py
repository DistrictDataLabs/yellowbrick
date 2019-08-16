# yellowbrick.download
# Downloads the example datasets for running the examples.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Created:  Wed May 18 11:54:45 2016 -0400
#
# Copyright (C) 2016 The sckit-yb developers
# For license information, see LICENSE.txt
#
# ID: download.py [1f73d2b] benjamin@bengfort.com $

"""
Downloads the example datasets for running the examples.
"""

##########################################################################
## Imports
##########################################################################

import os
import argparse

from yellowbrick.datasets import get_data_home
from yellowbrick.datasets.loaders import DATASETS
from yellowbrick.datasets.download import download_data
from yellowbrick.datasets.path import cleanup_dataset


##########################################################################
## Functions
##########################################################################


def download_all(data_home=None, replace=False):
    """
    Downloads all the example datasets to the data directory specified by
    ``get_data_home``. This function ensures that all datasets are available
    for use with the examples.
    """
    for _, meta in DATASETS.items():
        download_data(
            meta["url"], meta["signature"], data_home=data_home, replace=replace
        )

    print(
        "Downloaded {} datasets to {}".format(len(DATASETS), get_data_home(data_home))
    )


def cleanup_all(data_home=None):
    """
    Cleans up all the example datasets in the data directory specified by
    ``get_data_home`` either to clear up disk space or start from fresh.
    """
    removed = 0
    for name, meta in DATASETS.items():
        _, ext = os.path.splitext(meta["url"])
        removed += cleanup_dataset(name, data_home=data_home, ext=ext)

    print(
        "Removed {} fixture objects from {}".format(removed, get_data_home(data_home))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Yellowbrick data downloader",
        epilog="for troubleshooting please visit our GitHub issues",
    )
    parser.add_argument(
        "-c",
        "--cleanup",
        action="store_true",
        default=False,
        help="cleanup any existing datasets before download",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        default=False,
        help="prevent new data from being downloaded",
    )
    parser.add_argument(
        "-f",
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite any existing data with new download",
    )
    parser.add_argument(
        "data_home",
        default=None,
        nargs="?",
        help="specify the data download location or set $YELLOWBRICK_DATA",
    )

    args = parser.parse_args()

    if args.cleanup:
        cleanup_all(data_home=args.data_home)

    if not args.no_download:
        download_all(data_home=args.data_home, replace=args.overwrite)
