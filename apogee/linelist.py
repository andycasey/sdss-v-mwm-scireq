
import numpy as np
import os
from astropy.table import Table
from .config import config


def get_linelist(basename="linelist.vac.bestcopy"):

    path = os.path.join(config["APOGEE_DIR"], "linelist", basename)
    names = ("lam", "orggf", "newgf", "enewgf", "snewgf", "astgf", "sastgf", 
             "eastgf", "specid", "EP1", "J1", "EP1id", "EP2", "J2", "EP2id",
             "Rad", "sta", "vdW", "org", "unlte", "lnlte", "iso1", "hyp",
             "iso2", "isof", "hE1", "hE2", "F1", "note1", "S", "F2", "note2",
             "str", "aut", "g1", "g2", "ilam", "EW1", "EW2", "vdWorg")
    col_starts = (1, 11, 19, 27, 32, 35, 43, 47, 52, 60, 72, 77, 88,100,105,
        116,122,128,135,139,141,143,146,152,155,161,166,171,175,176,177,181,182,
        183,186,191,196,202,206,210)
    col_ends = (9, 17, 25, 30, 34, 41, 45, 50, 59, 71, 76, 87, 99,104,115,121,
        127,133,137,140,142,145,151,154,160,165,170,174,176,177,180,182,183,
        185,190,195,201,205,209,212)
    col_starts = tuple(np.array(col_starts) - 1)
    col_ends = tuple(np.array(col_ends) - 1)

    return Table.read(path, format="ascii.fixed_width_no_header", names=names,
        col_starts=col_starts, col_ends=col_ends)

