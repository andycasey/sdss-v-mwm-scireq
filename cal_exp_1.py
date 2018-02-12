
"""
Plot calibration visits.
"""

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from astropy.io import fits
from astropy.table import Table

import thecannon as tc
import thecannon.continuum

from apogee import config
from apogee.io import read_spectrum

from experiments import get_balanced_training_set, precision_from_repeat_calibration_visits
from plot_precision import  plot_precision_relative_to_aspcap

label_names = ["TEFF", "LOGG", "FE_H", "C_FE", "N_FE", "O_FE", "NA_FE",
               "MG_FE", "AL_FE", "SI_FE", "P_FE", "S_FE", "K_FE", "CA_FE",
               "TI_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]
label_names_for_balancing = ["TEFF", "LOGG", "FE_H", "NA_FE", "O_FE", "MG_FE",
                             "AL_FE"]


# ----------- #
train_kwds = dict(op_kwds=dict(factr=1e12, pgtol=1e-5))
test_kwds = dict()
show_kwds = dict(
    fitted=True,
    scatter_kwds=dict(visible=True),
    fill_between_kwds=dict(visible=True)
)
OUTPUT_PATH = "experiments/1/"
# ----------- #

if not os.path.exists(OUTPUT_PATH): os.mkdir(OUTPUT_PATH)


model_path = os.path.join(OUTPUT_PATH, "baseline.model")
if os.path.exists(model_path):
    model = tc.CannonModel.read(model_path)

else:
    vacuum_wavelengths, training_set_labels, training_set_flux, training_set_ivar \
        = get_balanced_training_set(label_names, label_names_for_balancing)

    vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)

    model = tc.CannonModel(
        training_set_labels, training_set_flux, training_set_ivar, vectorizer,
        dispersion=vacuum_wavelengths)

    model.train(**train_kwds)
    model.write(model_path, overwrite=True)


#visit_snr, combined_snr, visit_snr_labels, combined_snr_labels, apogee_ids \
results_path = os.path.join(OUTPUT_PATH, "calibration_precision_1.pkl")
if os.path.exists(results_path):
    with open(results_path, "rb") as fp:
        results = pickle.load(fp)

else:
    results = precision_from_repeat_calibration_visits(model)
    with open(results_path, "wb") as fp:
        pickle.dump(results, fp)


# Plot precision compared to ASPCAP values.
experiments = [
    # Label, model basename, result basename, kwds
    ["Baseline", model_path, results_path, show_kwds],
]
fig = plot_precision_relative_to_aspcap(experiments, label_names)

