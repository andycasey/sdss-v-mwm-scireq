

import numpy as np
import os

from .config import config

def element_filter(species, date=26112015):

    dir = os.path.join(
        config["APOGEE_DR14_DIR"], 
        "speclib/trunk/lib", 
        "filters_{}".format(date)
    )

    filter_path = os.path.join(dir, "{}_{}.filt".format(species, date))
    wavelength_path = os.path.join(dir, "wave.dat")

    raise a



def element_window(species, date=26112015):

    path = os.path.join(
        config["APOGEE_DIR"], 
        "speclib/trunk/lib", 
        "filters_{}".format(date),
        "{}.wind".format(species)
    )
    return np.loadtxt(path)
