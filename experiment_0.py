

"""
Experiment 0: Is The Cannon model used for APOGEE DR14 optimised correctly?
			  What label precision can be achieved from a polynomial (second-order)
			  model without censoring or regularisation?
"""

import numpy as np
import os
from astropy.io import fits
from astropy.table import Table

import thecannon as tc
import thecannon.continuum

from apogee import config
from apogee.io import read_spectrum


training_set_labels = Table.read(
	os.path.join(config["CANNON_DR14_DIR"], "apogee-dr14-giants-xh-censor-training-set.fits"))

# Load in fluxes and inverse variances.
N_pixels = 8575 # Number of pixels per APOGEE spectrum.
N_training_set_stars = len(training_set_labels)

training_set_flux = np.ones((N_training_set_stars, N_pixels))
training_set_ivar = np.zeros_like(training_set_flux)

continuum_kwds = dict(
    regions=[
        (15101, 15833), 
        (15834, 16454),
        (16455, 16999)
    ],
    continuum_pixels=np.loadtxt(os.path.join(
        config["CANNON_DR14_DIR"], "continuum_pixels.list"), dtype=int),
    L=1400, order=3, fill_value=1.0)


for i, star in enumerate(training_set_labels):

    # Load in the spectrum.
    vacuum_wavelength, flux, ivar, metadata = read_spectrum(
        star["TELESCOPE"], star["LOCATION_ID"], star["FILE"])

    # Continuum normalize.
    continuum, continuum_metadata = tc.continuum.sines_and_cosines(
        vacuum_wavelength, flux, ivar, **continuum_kwds)

    # Flatten arrays.
    flux, ivar, continuum = (flux.flatten(), ivar.flatten(), continuum.flatten())

    print(i)

raise a

label_names = ["TEFF", "LOGG", "FE_H"]
vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)

model = tc.CannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer)

model.train()

# 1-to-1.

# Which stars have repeat visits?

