

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
    os.path.join(config["CANNON_DR14_DIR"], 
    "apogee-dr14-giants-xh-censor-training-set.fits"))

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

# Original labels used by Holtz.
label_names = ["TEFF", "LOGG", "M_H", "ALPHA_M", "FE_H", "C_FE", "CI_FE", "N_FE",
               "O_FE", "NA_FE", "MG_FE", "AL_FE", "SI_FE", "P_FE", "S_FE", "K_FE",
               "CA_FE", "TI_FE", "TIII_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]
# The Cannon very responsibly gives warnings that:
# + [Fe/H] and [M/H] are highly correlated (rho = 1.0)
# + [Si/Fe] and [alpha/M] are highly correlated (rho = 0.93)
# + [Mg/Fe] and [alpha/M] are highly correlated (rho = 0.93)

# Therefore: removing [alpha/m] and [m/h]
label_names = ["TEFF", "LOGG", "FE_H", "C_FE", "CI_FE", "N_FE", "O_FE", "NA_FE", 
               "MG_FE", "AL_FE", "SI_FE", "P_FE", "S_FE", "K_FE", "CA_FE", 
               "TI_FE", "TIII_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]


vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)
raise a
model = tc.CannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer)

raise a
model.train()


# TODO: Save the one-to-one for this model
# TODO: Re-train models with TEFF, LOGG, M_H, ALPHA_M, and each element at a
#       time, and plot the spectral derivatives with respect to the main model.
#       Are they equivalent, or is it possible there is some bad training?
# TODO: Run the full model on all visits and plot dispersion as a function of 
#       S/N value. This will be our baseline.

raise a


oto_labels, oto_cov, oto_meta = model.test(training_set_flux, training_set_ivar)



for i, label_name in enumerate(vectorizer.label_names):

    x, y = (training_set_labels[label_name], oto_labels[:, i])

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = (min(limits), max(limits))
    ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.text(0.05, 0.90, "{:.2f} {:.2f}".format(np.mean(y-x), np.std(y-x)),
            transform=ax.transAxes)

    ax.set_title(label_name)


# Analyse all stars and calculate dispersion with respect to S/N.



