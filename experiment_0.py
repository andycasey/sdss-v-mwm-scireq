
import numpy as np
from astropy.io import fits
from astropy.table import Table

import thecannon as tc
from apogee.io import read_spectrum

training_set_labels = Table.read("apogee-dr14-giants-xh-censor-training-set.fits")

# Load in fluxes and inverse variances.
N_pixels_per_star = 3555
N_training_set_stars = len(training_set_labels)

training_set_flux = np.ones((N_training_set_stars, N_pixels_per_star))
training_set_ivar = np.zeros_like(training_set_flux)

for i, star in enumerate(training_set_labels):

    # Load in the spectrum.
    flux, ivar, metadata = read_spectrum(star["APSTAR_ID"])

    # Continuum normalize.


label_names = ["TEFF", "LOGG", "FE_H"]
vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)

model = tc.CannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer)

model.train()

# 1-to-1.

# Which stars have repeat visits?