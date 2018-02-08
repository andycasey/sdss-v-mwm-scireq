

__all__ = ["has_any_starflag", "has_all_starflag", "APOGEE_STARFLAG"]

import numpy as np
from collections import OrderedDict

APOGEE_STARFLAG = OrderedDict([
    (0, "BAD_PIXELS"),
    (1, "COMMISSIONING"),
    (2, "BRIGHT_NEIGHBOUR"),
    (3, "VERY_BRIGHT_NEIGHBOUR"),
    (4, "LOW_SNR"),
    (9, "PERSIST_HIGH"),
    (10, "PERSIST_MED"),
    (11, "PERSIST_LOW"),
    (12, "PERSIST_JUMP_POS"),
    (13, "PERSIST_JUMP_NEG"),
    (16, "SUSPECT_RV_COMBINATION"),
    (17, "SUSPECT_BROAD_LINES")
])




def _has_bitmasks(bitmask, flag_descriptions, flag_dictionary, operator):

    if isinstance(flag_descriptions, (str, )):
        flag_descriptions = [flag_descriptions]

    bitmask_integers = []
    for flag_description in flag_descriptions:
        if flag_description not in flag_dictionary.values():
            raise ValueError("invalid flag description '{}' (available: {})"\
                .format(flag_description, ", ".join(flag_dictionary.values())))

        bitmask_integers.extend(
            [k for k, v in flag_dictionary.items() if v == flag_description])


    return operator([(bitmask & 2**i) for i in bitmask_integers]).reshape(np.shape(bitmask))
    


def has_any_starflag(bitmask, flag_descriptions):
    return _has_bitmasks(bitmask, flag_descriptions, APOGEE_STARFLAG,
        lambda *x: np.any(x, axis=1))


def has_all_starflag(bitmask, flag_descriptions):
    return _has_bitmasks(bitmask, flag_descriptions, APOGEE_STARFLAG,
        lambda *x: np.all(x, axis=1))


