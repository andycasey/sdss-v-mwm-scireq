

# Do one-to-one for each regularised model. Record chi-squares and stddevs.


import logging
import numpy as np
import os
import pickle
from glob import glob


import thecannon as tc
import thecannon.restricted

from apogee.aspcap import element_window
from experiments import setup_training_set, precision_from_repeat_visits

# ---------------- #

OVERWRITE = False
OUTPUT_PATH = "experiments/4"

test_kwds = dict(op_kwds=dict(factr=1e12, pgtol=1e-5))
train_kwds = dict(threads=2, op_kwds=dict(factr=1e12, pgtol=1e-5))

label_names = ["TEFF", "LOGG", "FE_H", "C_FE", "N_FE", "O_FE", "NA_FE", 
               "MG_FE", "AL_FE", "SI_FE", "P_FE", "S_FE", "K_FE", "CA_FE", 
               "TI_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]

vacuum_wavelengths, training_set_labels, training_set_flux, training_set_ivar \
    = setup_training_set(full_output=True)


def get_path(regularization, restricted, suffix):
    prefix = "RestrictedModel" if restricted else "CannonModel"
    basename = "{}_regularized_10^{:.2f}_{}".format(
        prefix, np.log10(regularization), suffix)
    return os.path.join(OUTPUT_PATH, basename)
        


for regularization in np.logspace(-3, 5, 30):

    # Do the RestrictedModel first.
    result_path = get_path(regularization, True, "oto_results.pkl")
    meta_path = get_path(regularization, True, "meta.pkl")
    
    if os.path.exists(result_path) and os.path.exists(meta_path):
        print("Skipping regularization = {}".format(regularization))
        continue       


    model = tc.CannonModel.read(get_path(regularization, True, ".model"))
    oto_labels, oto_cov, oto_meta = model.test(
        training_set_flux, training_set_ivar, **test_kwds)

    with open(result_path, "wb") as fp:
        pickle.dump((oto_labels, oto_cov, oto_meta), fp, -1)

    # Calculate oto standard deviations and chi-squared.
    oto_chisq = (model(oto_labels) - training_set_flux)**2 * training_set_ivar

    meta = dict(
        sparsity=(np.abs(model.theta) < 1e-6).sum()/float(model.theta.size),
        oto_chisq=np.sum(oto_chisq),
        oto_label_means=dict(zip(
            model.vectorizer.label_names,
            np.nanmean(oto_labels - model.training_set_labels, axis=0)
        )),
        oto_label_sigmas=dict(zip(
            model.vectorizer.label_names,
            np.nanstd(oto_labels - model.training_set_labels, axis=0)
        ))
    )

    with open(meta_path, "wb") as fp:
        pickle.dump((meta, ), fp, -1)




    prefix = "CannonModel_regularized_10^{:.2f}".format(np.log10(regularization))
    model_path = os.path.join(OUTPUT_PATH, "{}.model".format(prefix))
    model = tc.CannonModel.read(model_path)

    oto_labels, oto_cov, oto_meta = model.test(
        training_set_flux, training_set_ivar, **test_kwds)

    result_path = os.path.join(OUTPUT_PATH, "oto_results_10^{:.2f}.pkl".format(np.log10(regularization)))
    with open(result_path, "wb") as fp:
        pickle.dump((oto_labels, oto_cov, oto_meta), fp, -1)
