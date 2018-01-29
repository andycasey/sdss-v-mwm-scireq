

"""
Sandbox 0: Can we restrict abundance coefficients?
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


FIGURE_PATH = ""

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
        star["TELESCOPE"], star["LOCATION_ID"], star["FILE"])

    # Continuum normalize.
    normalized_flux, normalized_ivar, continuum, meta = tc.continuum.normalize(
        vacuum_wavelength, flux, ivar, **continuum_kwds)

    training_set_flux[i, :] = normalized_flux
    training_set_ivar[i, :] = normalized_ivar

    print(i)


# Check regions to make sure we haven't got an old bug.
in_region = np.zeros(vacuum_wavelength.size, dtype=bool)
for start, end in continuum_kwds["regions"]:

    ok = (end >= vacuum_wavelength) * (vacuum_wavelength >= start)
    in_region[ok] = True

assert np.all(training_set_ivar[:, ~in_region] == 0)
assert np.all(training_set_flux[:, ~in_region] == 1)


label_names = ["TEFF", "LOGG", "M_H", "MG_FE", "AL_FE"]
vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)

restricted_model = tc.restricted.RestrictedCannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer,
    theta_bounds=dict(M_H=(None, 0), MG_FE=(None, 0), AL_FE=(None, 0)))
restricted_model.train()

original_model = tc.CannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer)
original_model.train()


fig, axes = plt.subplots(3, sharex=True, sharey=True)
axes[0].plot(vacuum_wavelength, original_model.theta.T[3], c='b')
axes[1].plot(vacuum_wavelength, original_model.theta.T[4], c='b')
axes[2].plot(vacuum_wavelength, original_model.theta.T[5], c='b')

axes[0].plot(vacuum_wavelength, restricted_model.theta.T[3], c='r')
axes[1].plot(vacuum_wavelength, restricted_model.theta.T[4], c='r')
axes[2].plot(vacuum_wavelength, restricted_model.theta.T[5], c='r')

axes[0].set_ylabel("M_H")
axes[1].set_ylabel("MG_FE")
axes[2].set_ylabel("AL_FE")

fig.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, "theta_coefficients.pdf"))



# Do one-to-one.
oto_orig_labels, oto_orig_cov, oto_orig_meta = original_model.test(
    training_set_flux, training_set_ivar)
oto_rest_labels, oto_rest_cov, oto_rest_meta = restricted_model.test(
    training_set_flux, training_set_ivar)


L = len(label_names)
A = int(np.ceil(L**0.5))
fig, axes = plt.subplots(A, A)
axes = np.array(axes).flatten()

for i, (ax, label_name) in enumerate(zip(axes, vectorizer.label_names)):

    x = training_set_labels[label_name]
    y_orig = oto_orig_labels[:, i]
    y_rest = oto_rest_labels[:, i]

    ax.scatter(x, y_orig, facecolor="b", alpha=0.5, s=1)
    ax.scatter(x, y_rest, facecolor="r", alpha=0.5, s=1)

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = (min(limits), max(limits))
    ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.text(0.05, 0.90, "{:.2f} {:.2f}".format(np.mean(y_orig-x), np.std(y_orig-x)),
            transform=ax.transAxes)
    ax.text(0.05, 0.75, "{:.2f} {:.2f}".format(np.mean(y_rest-x), np.std(y_rest-x)),
            transform=ax.transAxes)


    ax.set_title(label_name)

fig.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, "one_to_one.pdf"))

