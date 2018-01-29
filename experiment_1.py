
"""
Experiment 1: Can we make The Cannon model slightly more physically realistic
              by restricting the abundance coefficients to be negative, and
              removing abundance cross-terms from the vectorizer.
"""

import numpy as np
import os
from astropy.io import fits
from astropy.table import Table

import thecannon as tc
import thecannon.continuum
import thecannon.restricted

from apogee import config
from apogee.io import read_spectrum


# TODO: Train a new model using bounds on the abundance label coefficients.

# TODO: Train a new model without abundance cross-terms.

# TODO: Train a new model using bounds on the abundance label coefficients,
#       *and* without abundance cross-terms.

# Run analysis on individual visits and plot dispersion as a function of S/N.


# Restrict the vectorizer to only use cross-terms with TEFF and LOGG:
new_terms = \
    [t for t in vectorizer.get_human_readable_label_vector().split(" + ")[1:]
        if "*" not in t or ("TEFF" in t.split("*") or "LOGG" in t.split("*"))]
new_vectorizer = tc.vectorizer.PolynomialVectorizer(terms=new_terms)

