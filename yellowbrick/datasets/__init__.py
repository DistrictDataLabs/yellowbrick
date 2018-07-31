# yellowbrick.datasets
# Management utilities for Yellowbrick example datasets.
#
# Author:  Raul Peralta <raulpl25@gmail.com>
# Author:  Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Author:  Nathan Danielsen <ndanielsen@users.noreply.github.com>
# Created: Tue May 15 11:54:45 2018 -0400
#
# ID: __init__.py [] raulpl25@gmail.com $

"""
Management utilities for Yellowbrick example datasets.
"""

##########################################################################
## Imports
##########################################################################

from .loaders import load_concrete
from .loaders import load_energy
from .loaders import load_credit
from .loaders import load_occupancy
from .loaders import load_mushroom
from .loaders import load_hobbies
from .loaders import load_game
from .loaders import load_bikeshare
from .loaders import load_spam

from .path import get_data_home
