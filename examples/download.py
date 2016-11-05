#!/usr/bin/env python
# download
# Downloads the example datasets for running the examples.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
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

import os
import sys
import zipfile

try:
    import requests
except ImportError:
    print((
        "The requests module is required to download data --\n"
        "please install it with pip install requests."
    ))
    sys.exit(1)

##########################################################################
## Links to data sets
##########################################################################

OCCUPANCY = ('http://bit.ly/ddl-occupancy-dataset', 'occupancy.zip')
CREDIT    = ('http://bit.ly/ddl-credit-dataset', 'credit.xls')
CONCRETE  = ('http://bit.ly/ddl-concrete-data', 'concrete.xls')


def download_data(url, name, path='data'):
    if not os.path.exists(path):
        os.mkdir(path)

    response = requests.get(url)
    with open(os.path.join(path, name), 'w') as f:
        f.write(response.content)


def download_all(path='data'):
    for href, name in (OCCUPANCY, CREDIT, CONCRETE):
        download_data(href, name, path)

    # Extract the occupancy zip data
    z = zipfile.ZipFile(os.path.join(path, 'occupancy.zip'))
    z.extractall(os.path.join(path, 'occupancy'))


if __name__ == '__main__':
    path='data'
    download_all(path)
    print("Downloaded datasets to {}".format(os.path.abspath(path)))
