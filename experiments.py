
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table

import thecannon as tc
import thecannon.continuum

from apogee import config
from apogee.io import read_spectrum


def setup_training_set(filename="apogee-dr14-giants-xh-censor-training-set.fits"):

    training_set_labels = Table.read(
        os.path.join(config["CANNON_DR14_DIR"], filename))

    # Load in fluxes and inverse variances.
    N_pixels = 8575 # Number of pixels per APOGEE spectrum.
    N_training_set_stars = len(training_set_labels)

    training_set_flux = np.ones((N_training_set_stars, N_pixels))
    training_set_ivar = np.zeros_like(training_set_flux)


    continuum_kwds = dict(
        regions=[
            (15140, 15812),
            (15857, 16437),
            (16472, 16960)
        ],
        continuum_pixels=np.loadtxt(os.path.join(
            config["CANNON_DR14_DIR"], "continuum_pixels.list"), dtype=int),
        L=1400, order=3, fill_value=np.nan)

    for i, star in enumerate(training_set_labels):

        # Load in the spectrum.
        vacuum_wavelength, flux, ivar, metadata = read_spectrum(
            star["TELESCOPE"], star["FIELD"], star["LOCATION_ID"], star["FILE"])

        # Continuum normalize.
        normalized_flux, normalized_ivar, continuum, meta = tc.continuum.normalize(
            vacuum_wavelength, flux, ivar, **continuum_kwds)

        training_set_flux[i, :] = normalized_flux
        training_set_ivar[i, :] = normalized_ivar

        print(i)


    return (training_set_labels, training_set_flux, training_set_ivar)


