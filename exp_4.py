
"""
Using our balanced training set:

[X] Train a restricted Cannon model without cross-terms, and censor the windows
    based on the line list.

[X] Plot theta.

[X] Plot one-to-one.

[X] Calculate precision as a function of SNR.

[ ] See how well we do on globular clusters.
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
import thecannon.restricted

from apogee import config, linelist
from apogee.io import read_spectrum

from experiments import get_balanced_training_set, precision_from_repeat_visits

label_names = ["TEFF", "LOGG", "FE_H", "C_FE", "CI_FE", "N_FE", "O_FE", "NA_FE",
               "MG_FE", "AL_FE", "SI_FE", "P_FE", "S_FE", "K_FE", "CA_FE",
               "TI_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]
label_names_for_balancing = ["TEFF", "LOGG", "FE_H", "NA_FE", "O_FE", "MG_FE",
                             "AL_FE"]

#vacuum_wavelengths, training_set_labels, training_set_flux, training_set_ivar \
#    = get_balanced_training_set(label_names, label_names_for_balancing)
with open("experiments/2/training_set.pkl", "rb") as fp:
    vacuum_wavelengths, training_set_labels, training_set_flux, training_set_ivar \
        = pickle.load(fp)


# ----------- #
train_kwds = dict(op_kwds=dict(factr=1e12, pgtol=1e-5))
test_kwds = dict()
OUTPUT_PATH = "experiments/4/"
# ----------- #

if not os.path.exists(OUTPUT_PATH): os.mkdir(OUTPUT_PATH)


transitions = linelist.get_linelist()

window = 0.5 # Angstroms surrounding each line.
elements = [ln.split("_")[0] for ln in label_names if "_" in ln]
censors = {}
for label_name in label_names:
    if "_" not in label_name: continue

    element = label_name.split("_")[0]

    Z = atomic_number(element)

    censors[label_name] = np.zeros(vacuum_wavelengths.size, dtype=bool)

    match = (transitions["specid"].astype(int) == Z)
    for wavelength in transitions["lam"][match]:
        wl = wavelength * 10 # to angstroms.
        censors[label_name] += ((wl + window) >= vacuum_wavelengths) * (vacuum_wavelengths >= (wl - window))

    censors[label_name] = censors[label_name]


vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)
# Restrict the vectorizer to only use cross-terms with TEFF and LOGG:
new_terms = \
    [t for t in vectorizer.get_human_readable_label_vector().split(" + ")[1:]
        if "*" not in t or ("TEFF" in t.split("*") or "LOGG" in t.split("*"))]
vectorizer = tc.vectorizer.PolynomialVectorizer(terms=new_terms)


model = tc.restricted.RestrictedCannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer,
    dispersion=vacuum_wavelengths, censors=censors,
    theta_bounds=dict([(ln, (None, 0)) for ln in label_names if ln.endswith("_FE")]))

model.train(**train_kwds)
model.write(os.path.join(OUTPUT_PATH, "restricted_wo_ct_censored.model"), overwrite=True)


# Plot theta.
fig = tc.plot.theta(model, indices=np.arange(len(label_names) + 1))
fig.set_figheight(20)
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(os.path.join(OUTPUT_PATH, "baseline_wo_ct_censored_theta.pdf"), dpi=300)


# Do one-to-one.
oto_labels, oto_cov, oto_meta = model.test(training_set_flux, training_set_ivar,
                                           **test_kwds)

fig = tc.plot.one_to_one(model, oto_labels)
fig.savefig(os.path.join(OUTPUT_PATH, "baseline_wo_ct_censored_one_to_one.pdf"))


# Run the model on all visits and plot dispersion as a function of S/N value.
snr, combined_snr, label_difference, filename = precision_from_repeat_visits(
    model, N_comparisons=1000, test_kwds=test_kwds)

with open(os.path.join(OUTPUT_PATH, "baseline_wo_ct_censored_precision_snr.pkl"), "wb") as fp:
    pickle.dump((snr, combined_snr, label_difference, filename), fp)

