

"""
Experiment 0: Is The Cannon model used for APOGEE DR14 optimised correctly?
              What label precision can be achieved from a polynomial (second-order)
              model without censoring or regularisation?
"""

import os
from astropy.table import Table
from apogee.io import get_spectrum_path
from apogee import config


all_stars = Table.read(
    os.path.join(config["APOGEE_DR14_DIR"], "allStar-l31c.2.fits"))


missing_files = []
unique_files = []
for i, star in enumerate(all_stars):

    path = get_spectrum_path(
        star["TELESCOPE"], star["FIELD"], star["LOCATION_ID"], star["FILE"])
    unique_files.append(path)

    if path is None or not os.path.exists(path):
        missing_files.append(path)
        print("Missing #{}: {}".format(len(missing_files), path))

N_unique = len(set(unique_files))
N_missing = len(set(missing_files))

