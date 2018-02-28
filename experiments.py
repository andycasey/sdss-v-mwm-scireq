
import logging
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table

import thecannon as tc
import thecannon.continuum

from apogee import config, bitmasks
from apogee.io import read_spectrum, read_aspcapstar_spectrum


continuum_kwds = dict(
    regions=[
        (15145.0, 15807.0),
        (15862.0, 16432.0),
        (16480.0, 16952.0)
    ],
    continuum_pixels=np.loadtxt(os.path.join(
        config["CANNON_DR14_DIR"], "continuum_pixels.list"), dtype=int),
    L=1400, order=3, fill_value=np.nan)

test_kwds = dict(initial_labels=None, threads=1)
train_kwds = dict(op_method="l_bfgs_b", op_kwds=dict(factr=1e12, pgtol=1e-5),
                  threads=1)


def setup_training_set(filename="apogee-dr14-giants-training-set.fits",
    full_output=True, **kwargs):

    kwds = dict()
    kwds.update(continuum_kwds)
    kwds.update(kwargs)

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

        training_set_flux[i, :] = flux
        training_set_ivar[i, :] = ivar

        print(i)

    # Continuum normalize.
    normalized_flux, normalized_ivar, continuum, meta = tc.continuum.normalize(
        vacuum_wavelength, training_set_flux, training_set_ivar,
        **kwds)

    training_set_flux = normalized_flux
    training_set_ivar = normalized_ivar

    if full_output:
        return (vacuum_wavelength, training_set_labels, training_set_flux,
            training_set_ivar)
    return (training_set_labels, training_set_flux, training_set_ivar)



def training_set_data(stars, **kwargs):

    kwds = continuum_kwds.copy()
    kwds.update(kwargs)

    # Load in fluxes and inverse variances.
    N_pixels = 8575 # Number of pixels per APOGEE spectrum.
    N_training_set_stars = len(stars)

    training_set_flux = np.ones((N_training_set_stars, N_pixels))
    training_set_ivar = np.zeros_like(training_set_flux)

    print("Number of star in training set: {}".format(N_training_set_stars))

    for i, star in enumerate(stars):

        # Load in the spectrum.
        try:
            vacuum_wavelength, flux, ivar, metadata = read_spectrum(
                star["TELESCOPE"], star["FIELD"], star["LOCATION_ID"], star["FILE"])
        except:
            logging.warn("Error on star {}".format(i))
            continue


        # Continuum normalize.
        normalized_flux, normalized_ivar, continuum, meta \
            = tc.continuum.normalize(vacuum_wavelength, flux, ivar, **kwds)

        training_set_flux[i, :] = normalized_flux
        training_set_ivar[i, :] = normalized_ivar

        print(i)

    pixel_is_used = np.zeros(N_pixels, dtype=bool)
    for start, end in kwds["regions"]:
        region_mask = (end >= vacuum_wavelength) * (vacuum_wavelength >= start)
        pixel_is_used[region_mask] = True

    training_set_flux[:, ~pixel_is_used] = 1.0
    training_set_ivar[:, ~pixel_is_used] = 0.0

    return (vacuum_wavelength, training_set_flux, training_set_ivar)




def setup_training_set_from_aspcap(
    filename="apogee-dr14-giants-xh-censor-training-set.fits",
    full_output=True, return_model_spectrum=True, continuum_normalize=True):

    training_set_labels = Table.read(
        os.path.join(config["CANNON_DR14_DIR"], filename))

    # Load in fluxes and inverse variances.
    N_pixels = 8575 # Number of pixels per APOGEE spectrum.
    N_training_set_stars = len(training_set_labels)

    training_set_flux = np.ones((N_training_set_stars, N_pixels))
    training_set_ivar = np.zeros_like(training_set_flux)


    for i, star in enumerate(training_set_labels):

        # Load in the spectrum.
        try:
            vacuum_wavelength, flux, ivar, metadata = read_aspcapstar_spectrum(
                star["LOCATION_ID"], star["APOGEE_ID"], return_model_spectrum=return_model_spectrum)
        except:
            print("FAILED ON {}".format(i))
            continue

        if not continuum_normalize:
            normalized_flux, normalized_ivar = flux, ivar
        else:
            normalized_flux, normalized_ivar, continuum, meta = tc.continuum.normalize(
                vacuum_wavelength, flux, ivar, **continuum_kwds)

        training_set_flux[i, :] = normalized_flux
        training_set_ivar[i, :] = normalized_ivar

        print(i)

    if full_output:
        return (vacuum_wavelength, training_set_labels, training_set_flux,
            training_set_ivar)
    return (training_set_labels, training_set_flux, training_set_ivar)



def generate_calibration_visit_comparison():

    calibrations = fits.open(os.path.join(
        config["APOGEE_DR14_DIR"], "l31c", "l31c.1", "allCal-l31c.1.fits"))

    stars = fits.open(os.path.join(
        os.path.join(config["APOGEE_DR14_DIR"], "allStar-l31c.2.fits")))[1].data

    for i, apogee_id in enumerate(set(calibrations[1].data["APOGEE_ID"])):

        try:
            star = stars[np.where(stars["APOGEE_ID"] == apogee_id)[0][0]]

            vacuum_wavelength, flux, ivar, metadata = read_spectrum(
                star["TELESCOPE"], star["FIELD"], star["LOCATION_ID"], star["FILE"],
                combined=False)
        except:
            continue


        if flux.size == 0 or not np.any(ivar > 0):
            continue

        try:
            normalized_flux, normalized_ivar, _, __ = tc.continuum.normalize(
                vacuum_wavelength, flux, ivar, **continuum_kwds)

        except:
            continue


        yield (vacuum_wavelength, normalized_flux, normalized_ivar, star,
            metadata)




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



def aspcap_precision_from_repeat_calibration_visits(label_names):
    calibrations = fits.open(os.path.join(
        config["APOGEE_DR14_DIR"], "l31c", "l31c.1", "allCal-l31c.1.fits"))

    stars = fits.open(os.path.join(
        os.path.join(config["APOGEE_DR14_DIR"], "allStar-l31c.2.fits")))[1].data

    visit_snr = []
    combined_snr = []
    combined_snr_labels = []
    visit_snr_labels = []
    apogee_ids = []

    unique_apogee_ids = set(calibrations[1].data["APOGEE_ID"])
    N = len(unique_apogee_ids)

    for i, apogee_id in enumerate(unique_apogee_ids):

        print(i, N, apogee_id)
        try:
            star = stars[np.where(stars["APOGEE_ID"] == apogee_id)[0][0]]
        except:
            logging.warn("Cannot find apogee id {}".format(apogee_id))
            continue

        high_snr_labels = np.array([star[ln] for ln in label_names])

        match = np.where(calibrations[1].data["APOGEE_ID"] == apogee_id)[0]

        N_visits = len(match)
        visit_snr.extend(calibrations[1].data["SNR"][match])
        combined_snr.extend([star["SNR"]] * N_visits)
        combined_snr_labels.extend(
            np.tile(high_snr_labels, N_visits).reshape((N_visits, -1)))

        apogee_ids.extend([apogee_id] * N_visits)

        elem_symbol = ['C', 'CI', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
        'K', 'Ca', 'Ti', 'TiII', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Ge',
        'Ce', 'Rb', 'Y', 'Nd']
        elem_symbol = [each.upper() for each in elem_symbol]


        foo = []
        for label_name in label_names:

            if label_name == "TEFF":
                x = calibrations[1].data["PARAM"][match, 0]
            elif label_name == "LOGG":
                x = calibrations[1].data["PARAM"][match, 1]
            else:
                element, relative_to = label_name.split("_")
                elem_index = elem_symbol.index(element)

                if relative_to == "FE":
                    fe_index = elem_symbol.index("FE")
                    x = calibrations[1].data["X_H"][match, elem_index] \
                      - calibrations[1].data["X_H"][match, fe_index]

                else:
                    x = calibrations[1].data["X_H"][match, elem_index]

            foo.append(x)

        visit_snr_labels.extend(np.vstack(foo).T)


    visit_snr = np.array(visit_snr)
    apogee_ids = np.array(apogee_ids)
    combined_snr = np.array(combined_snr)
    combined_snr_labels = np.array(combined_snr_labels)
    visit_snr_labels = np.array(visit_snr_labels)

    return (visit_snr, combined_snr, visit_snr_labels, combined_snr_labels,
        apogee_ids)



def precision_from_repeat_calibration_visits(model, N_comparisons=None,
    test_kwds=None):

    test_kwds = {} if test_kwds is None else test_kwds

    K = 0
    visit_snr = []
    combined_snr = []
    aspcap_combined_snr_labels = []
    visit_snr_labels = []
    apogee_ids = []

    for comparison in generate_calibration_visit_comparison():

        vacuum_wavelength, flux, ivar, star, metadata = comparison

        try:
            visit_labels, _, __ = model.test(flux, ivar, **test_kwds)

        except:
            logging.warn("Failed on comparison:")
            continue

        aspcap_labels = np.array([star[ln] for ln in model.vectorizer.label_names])

        N_visits = flux.shape[0]
        visit_snr.extend(metadata["snr_visits"])
        combined_snr.extend([star["SNR"]] * N_visits)
        aspcap_combined_snr_labels.extend(
            np.tile(aspcap_labels, N_visits).reshape((N_visits, -1)))
        visit_snr_labels.extend(visit_labels)
        apogee_ids.extend([star["APOGEE_ID"]] * N_visits)

        K += N_visits
        print("Number of comparisons so far: {}".format(K))

        if N_comparisons is not None and K >= N_comparisons: break

    visit_snr = np.array(visit_snr)
    combined_snr = np.array(combined_snr)
    aspcap_combined_snr_labels = np.array(aspcap_combined_snr_labels)
    visit_snr_labels = np.array(visit_snr_labels)
    apogee_ids = np.array(apogee_ids)

    return (visit_snr, combined_snr, visit_snr_labels, aspcap_combined_snr_labels,
        apogee_ids)



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


def get_globular_cluster_candidates(globular_cluster_fields=None,
    membership_criteria=None, require_labels=None, **kwargs):


    stars = Table.read(os.path.join(config["APOGEE_DR14_DIR"], "allStar-l31c.2.fits"))

    if membership_criteria is None:
        membership_criteria = {}

    if globular_cluster_fields is None:
        globular_cluster_fields = ('M12-N', 'M13', 'M15', 'M2', 'M3', 'M35N2158',
            'M5', 'M53', 'M54SGRC1', 'M5PAL5', 'M67', 'M71', 'M92', 'N1333', 'N188',
            'N2243', 'N2420', 'N4147', 'N5466', 'N5634SGR2', 'N6229', 'N6791',
            'N6819', 'N7789')


    is_gc_candidate = np.in1d(
        stars["FIELD"],
        ["{0:16s}".format(each) for each in globular_cluster_fields]
    )

    # Exclude things that ASPCAP did not find a TEFF/LOGG for.
    is_gc_candidate *= (stars["TEFF"] > 0) \
                    *  (stars["LOGG"] > -5000) \
                    *  (stars["FE_H"] > -5000)
    if require_labels is not None:
        for each in require_labels:
            is_gc_candidate *= (stars[each] > -5000)

    N_stars = sum(is_gc_candidate)

    print("{} candidates found in fields {}".format(N_stars,
        ", ".join(globular_cluster_fields)))



    kwds = continuum_kwds.copy()
    kwds.update(kwargs)

    # Load in fluxes and inverse variances.
    N_pixels = 8575 # Number of pixels per APOGEE spectrum.

    gc_candidate_flux = np.ones((N_stars, N_pixels))
    gc_candidate_ivar = np.zeros_like(gc_candidate_flux)

    gc_candidates = stars[is_gc_candidate]


    for i, star in enumerate(gc_candidates):

        # Load in the spectrum.
        try:
            vacuum_wavelength, flux, ivar, metadata = read_spectrum(
                star["TELESCOPE"], star["FIELD"], star["LOCATION_ID"], star["FILE"])
        except:
            logging.warn("Error on star {}".format(i))
            continue


        # Continuum normalize.
        normalized_flux, normalized_ivar, continuum, meta \
            = tc.continuum.normalize(vacuum_wavelength, flux, ivar, **kwds)

        gc_candidate_flux[i, :] = normalized_flux
        gc_candidate_ivar[i, :] = normalized_ivar

        print(i)

    pixel_is_used = np.zeros(N_pixels, dtype=bool)
    for start, end in kwds["regions"]:
        region_mask = (end >= vacuum_wavelength) * (vacuum_wavelength >= start)
        pixel_is_used[region_mask] = True

    gc_candidate_flux[:, ~pixel_is_used] = 1.0
    gc_candidate_ivar[:, ~pixel_is_used] = 0.0

    return (vacuum_wavelength, gc_candidates, gc_candidate_flux, gc_candidate_ivar)





def aspcap_precision_from_repeat_visits(label_names, N_comparisons=None,
    combined_filename="allStar-l31c.2.fits",
    visit_filename="allVisit-l31c.2.fits"):

    image_visits = fits.open(os.path.join(config["APOGEE_DR14_DIR"], visit_filename))
    combined_visits = fits.open(os.path.join(config["APOGEE_DR14_DIR"], combined_filename))

    visit_snr = []
    combined_snr = []
    visit_labels_difference = []
    filenames = []

    K = 0
    for i, target_id in enumerate(combined_visits[1].data["TARGET_ID"]):

        match = image_visits[1].data["TARGET_ID"] == target_id
        N_visits = sum(match)
        if 2 > N_visits:
            continue

        combined_labels = np.array([combined_visits[1].data[ln][i] for ln in label_names])
        visit_snr.extend(image_visits[1].data["SNR"][match])
        combined_snr.extend([combined_visits[1].data["SNR"][i]] * N_visits)

        visit_labels = np.array([image_visits[1].data[ln][match] for ln in label_names])
        visit_label_difference.extend(visit_labels - combined_labels)

        filenames.extend(image_visits[1].data["FILENAME"])

        K += N_visits
        print("Number of comparisons so far: {}".format(K))

        if N_comparisons is not None and K >= N_comparisons:
            break


    filenames = np.array(filenames)
    visit_snr = np.array(visit_snr)
    combined_snr = np.array(combined_snr)
    visit_label_difference = np.array(visit_label_difference)

    image_visits.close()
    combined_visits.close()

    del image_visits, combined_visits

    return (visit_snr, combined_snr, visit_label_difference, filenames)




    # Find stars that are the same.


def get_balanced_training_set(label_names, label_names_for_balancing,
    snr_min=100, max_aspcap_chi2=5, bad_starflags=None, N_bins=10,
    random_seed=42, no_balance=0):

    if bad_starflags is None:
        bad_starflags =  [
            "PERSIST_HIGH", "PERSIST_JUMP_POS", "PERSIST_JUMP_NEG",
            "SUSPECT_BROAD_LINES", "SUSPECT_RV_COMBINATION",
            "VERY_BRIGHT_NEIGHBOUR", "COMMISSIONING", "BAD_PIXELS",
        ]

    stars = Table.read(
        os.path.join(config["APOGEE_DR14_DIR"], "allStar-l31c.2.fits"))

    validation_set = (stars["SNR"] > snr_min) \
                    * (stars["ASPCAPFLAG"] == 0) \
                    * (stars["ASPCAP_CHI2"] < max_aspcap_chi2) \
                    * ~bitmasks.has_any_starflag(
                        stars["STARFLAG"], bad_starflags)


    m, c = (2.6/1100, -7.70)
    logg_criteria = stars["TEFF"] * m + c
    validation_set *= (stars["LOGG"] <= logg_criteria)

    # We want stars with abundances.....
    # We can get the other abundances, but these are the ones we want to verify
    # our model with.

    original_validation_set = validation_set.copy()
    for label_name in label_names:

        removed = (stars[label_name][original_validation_set] < -5000)
        print("We removed {} good stars due to missing {}".format(
            sum(removed), label_name))

        validation_set *= (stars[label_name] > -5000)

        if "{}_FLAG".format(label_name) in stars.dtype.names:
            assert not any(stars["{}_FLAG".format(label_name)] & 2**16)



    unbalanced_data = np.array([stars[ln][validation_set] for ln in label_names_for_balancing]).T

    H, bins = np.histogramdd(unbalanced_data, N_bins)
    indices = np.array([np.digitize(c, b) for c, b in zip(unbalanced_data.T, bins)]) - 1

    assert np.sum(np.all(indices.T == np.array(np.where(H == H.max())).flatten(), axis=1)) == H.max()

    """
    Here we will just take 1 star from every multidimensional bin that has a count
    of at least 1. That gives us some 5000 stars.
    If you want a larger sample size then you may need to over-sample some bins,
    or accept that you will have a *slightly* "unbalanced" data set (but far more
    balanced than what it was to begin with).
    """

    np.random.seed(random_seed)

    use_bins = np.array(np.where(H >= 1)).T
    in_training_set = np.zeros(len(unbalanced_data), dtype=bool)

    if no_balance > 0:
        in_training_set[np.random.choice(np.arange(len(in_training_set)), no_balance, replace=False)] = True

    else:

        for i, bin_indices in enumerate(use_bins):

            star_indices = np.where(np.all(indices.T == bin_indices, axis=1))[0]
            print(i, star_indices.size)

            if 1 > star_indices.size:
                print("What the fuck?")
                continue

            in_training_set[np.random.choice(star_indices, size=1)] = True


    star_indices = np.where(validation_set)[0][in_training_set]
    training_set = np.zeros(len(stars), dtype=bool)
    training_set[star_indices] = True

    # Construct training set.
    # Construct training set.
    training_set_labels = stars[training_set]
    vacuum_wavelengths, training_set_flux, training_set_ivar \
        = training_set_data(training_set_labels, **continuum_kwds)

    return (vacuum_wavelengths, training_set_labels, training_set_flux, training_set_ivar)
