
"""
Experiment 5: Train using ASPCAP-best-fitting spectra, and run using ASPCAP-best
-fitting spectra.
"""

import logging
import numpy as np
import os
import pickle

import thecannon as tc
import thecannon.restricted

from apogee.aspcap import element_window
from experiments import setup_training_set_from_aspcap, setup_training_set, precision_from_repeat_visits

# ---------------- #

OVERWRITE = True
OUTPUT_PATH = "experiments/6"

test_kwds = dict()
train_kwds = dict(threads=2, op_kwds=dict(factr=1e12, pgtol=1e-5))

label_names = ["TEFF", "LOGG", "FE_H"]

# ---------------- #

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

#vacuum_wavelengths, training_set_labels, training_set_flux, training_set_ivar \
#    = setup_training_set(full_output=True)

# Use ASPCAP best-fitting spectra to train the model.
vacuum_wavelengths, training_set_labels, training_set_flux, training_set_ivar \
    = setup_training_set_from_aspcap(full_output=True,
        return_model_spectrum=True, continuum_normalize=False)

vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)

prefix = os.path.join(OUTPUT_PATH, "aspcap_trained")
model_path = "{}.model".format(prefix)

model = tc.CannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer)
model.train(**train_kwds)
model.write(model_path, overwrite=OVERWRITE)


# Do one-to-one.
oto_labels, oto_cov, oto_meta = model.test(
    training_set_flux, training_set_ivar, **test_kwds)

fig = tc.plot.one_to_one(model, oto_labels)
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig("{}_oto.pdf".format(prefix), dpi=300)


# Plot theta.
fig = tc.plot.theta(model, indices=np.arange(len(label_names) + 1))
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig("{}_theta.pdf".format(prefix),
            dpi=300)

# Now apply this to the actual data.
_, __, observed_flux, observed_ivar = setup_training_set_from_aspcap(
    full_output=True, return_model_spectrum=False, continuum_normalize=False)

oto_labels_obs, oto_cov_obs, oto_meta_obs = model.test(
    observed_flux, observed_ivar, **test_kwds)

fig = tc.plot.one_to_one(model, oto_labels_obs)
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig("{}_oto_obs.pdf".format(prefix), dpi=300)


raise a

# Run the model on all visits and plot dispersion as a function of S/N value.
snr, combined_snr, label_difference, filename = precision_from_repeat_visits(
    model, N_comparisons=10000, test_kwds=test_kwds)

with open(os.path.join(OUTPUT_PATH, "RestrictedModel_with_aspcap_windows_precision_snr.pkl"), "wb") as fp:
    pickle.dump((snr, combined_snr, label_difference, filename), fp)


