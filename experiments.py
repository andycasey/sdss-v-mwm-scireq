
import logging
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table

import thecannon as tc
import thecannon.continuum

from apogee import config
from apogee.io import read_spectrum


# Useful and common keywords.
continuum_kwds = dict(
    regions=[
        (15140, 15812),
        (15857, 16437),
        (16472, 16960)
    ],
    continuum_pixels=np.loadtxt(os.path.join(
        config["CANNON_DR14_DIR"], "continuum_pixels.list"), dtype=int),
    L=1400, order=3, fill_value=np.nan)

test_kwds = dict(initial_labels=None, threads=1)
train_kwds = dict(op_method="l_bfgs_b", op_kwds=dict(factr=1e12, pgtol=1e-5),
                  threads=1)


def setup_training_set(filename="apogee-dr14-giants-xh-censor-training-set.fits"):

    training_set_labels = Table.read(
        os.path.join(config["CANNON_DR14_DIR"], filename))

    # Load in fluxes and inverse variances.
    N_pixels = 8575 # Number of pixels per APOGEE spectrum.
    N_training_set_stars = len(training_set_labels)

    training_set_flux = np.ones((N_training_set_stars, N_pixels))
    training_set_ivar = np.zeros_like(training_set_flux)


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


def generate_individual_visit_comparison(filename, randomize=True, random_seed=42):

    # Load the allStar file.

    # For each star, load the spectrum.

    # Send back the stacked spectrum and individual visits.


    # Load individual visits from

    stars = fits.open(os.path.join(
        os.path.join(config["APOGEE_DR14_DIR"], filename)))[1].data

    # Apply QC.
    print("HACK MAGIC APPLY QC")
    ok = (stars["TEFF"] > 0) * (stars["LOGG"] > -5000)
    stars = stars[ok]

    N = len(stars)
    if randomize:
        np.random.seed(random_seed)
        indices = np.random.choice(N, replace=False, size=N)
    else:
        indices = np.arange(N)

    for i, star in enumerate(stars[indices]):

        vacuum_wavelength, flux, ivar, metadata = read_spectrum(
            star["TELESCOPE"], star["FIELD"], star["LOCATION_ID"], star["FILE"],
            combined=False)

        if flux.size == 0 or not np.any(ivar > 0):
            # Only single visit, or S/N super high/low.
            continue

        _, combined_flux, combined_ivar, combined_metadata = read_spectrum(
            star["TELESCOPE"], star["FIELD"], star["LOCATION_ID"], star["FILE"],
            combined=True)


        try:
            normalized_flux, normalized_ivar, _, __ = tc.continuum.normalize(
                vacuum_wavelength, flux, ivar, **continuum_kwds)

            normalized_combined_flux, normalized_combined_ivar, _, __ = \
                tc.continuum.normalize(vacuum_wavelength, combined_flux,
                    combined_ivar, **continuum_kwds)

        except:
            continue

        # Create a comparison to add.
        yield (normalized_combined_flux, normalized_combined_ivar,
            normalized_flux, normalized_ivar, metadata)



def precision_from_repeat_visits(model, N_comparisons=None, test_kwds=None, 
    filename="allStar-l31c.2.fits", randomize=True, random_seed=42):


    test_kwds = {} if test_kwds is None else test_kwds

    K = 0
    visit_snr = []
    combined_snr = []
    visit_label_difference = []
    filenames = []

    for comparison in generate_individual_visit_comparison(filename,
        randomize=randomize, random_seed=random_seed):

        combined_flux, combined_ivar, visit_flux, visit_ivar, meta = comparison

        try:
            combined_labels, _, __ = model.test(combined_flux, combined_ivar, 
                                                **test_kwds)

            visit_labels, _, __ = model.test(visit_flux, visit_ivar, **test_kwds)

        except:
            logging.warn("Failed on comparison:")
            continue

        N_visits = visit_flux.shape[0]
        visit_snr.extend(meta["snr_visits"])
        combined_snr.extend([meta["snr_combined"]] * N_visits)
        filenames.extend([meta["filename"]] * N_visits)
        visit_label_difference.extend(visit_labels - combined_labels)
        
        K += N_visits

        print("Number of comparisons so far: {}".format(K))

        if N_comparisons is not None and K >= N_comparisons:
            break

    filenames = np.array(filenames)
    visit_snr = np.array(visit_snr)
    combined_snr = np.array(combined_snr)
    visit_label_difference = np.array(visit_label_difference)

    return (visit_snr, combined_snr, visit_label_difference, filenames)