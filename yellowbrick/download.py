#!/usr/bin/env python
# yellowbrick.download
# Downloads the example datasets for running the examples.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Wed May 18 11:54:45 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: download.py [1f73d2b] benjamin@bengfort.com $

"""
Downloads the example datasets for running the examples.
"""

##########################################################################
## Imports
##########################################################################

import argparse

from yellowbrick.datasets import get_data_home
from yellowbrick.datasets.loaders import DATASETS
from yellowbrick.datasets.download import download_data


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
            meta['url'], meta['signature'], data_home=data_home, replace=replace
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Yellowbrick data downloader",
        epilog="for troubleshooting please visit our GitHub issues"
    )
    parser.add_argument(
        '-f', '--overwrite', action='store_true', default=False,
        help="overwrite any existing data with new download",
    )
    parser.add_argument(
        "data_home", default=None, nargs="?",
        help="specify the data download location or set $YELLOWBRICK_DATA",
    )

    args = parser.parse_args()

    download_all(data_home=args.data_home, replace=args.overwrite)
    print(
        "Downloaded {} datasets to {}".format(
            len(DATASETS), get_data_home(args.data_home)
    ))
