
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
from experiments import setup_training_set_from_aspcap, precision_from_repeat_visits

# ---------------- #

OVERWRITE = True
OUTPUT_PATH = "experiments/5"

test_kwds = dict()
train_kwds = dict(threads=2, op_kwds=dict(factr=1e12, pgtol=1e-5))

label_names = ["TEFF", "LOGG", "FE_H", "C_FE", "N_FE", "O_FE", "NA_FE", 
               "MG_FE", "AL_FE", "SI_FE", "P_FE", "S_FE", "K_FE", "CA_FE", 
               "TI_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]

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

model = tc.restricted.RestrictedCannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer)
model.train(**train_kwds)
model.write(model_path, overwrite=OVERWRITE)


# Do one-to-one.
oto_labels, oto_cov, oto_meta = model.test(
    training_set_flux, training_set_ivar, **test_kwds)

fig = tc.plot.one_to_one(model, oto_labels)
fig.set_figheight(40)
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig("{}_oto2.pdf".format(prefix), dpi=300)



# Run the model on all visits and plot dispersion as a function of S/N value.
# THIS IS WRONG THING TO DO BUT WE DO IT FOR FUN.
snr, combined_snr, label_difference, filename = precision_from_repeat_visits(
    model, N_comparisons=10000, test_kwds=test_kwds)

with open(os.path.join(OUTPUT_PATH, "aspcap_trained_snr_precision.pkl"), "wb") as fp:
    pickle.dump((snr, combined_snr, label_difference, filename), fp)






prefix = os.path.join(OUTPUT_PATH, "aspcap_restricted_trained")
model_path = "{}.model".format(prefix)

model = tc.restricted.RestrictedCannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer,
    theta_bounds=dict([(ln, (None, 0)) for ln in label_names if ln.endswith("_FE")]))
model.train(**train_kwds)
model.write(model_path, overwrite=OVERWRITE)


# Do one-to-one.
oto_labels, oto_cov, oto_meta = model.test(
    training_set_flux, training_set_ivar, **test_kwds)

fig = tc.plot.one_to_one(model, oto_labels)
fig.set_figheight(40)
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig("{}_oto2.pdf".format(prefix), dpi=300)


# Run the model on all visits and plot dispersion as a function of S/N value.
# THIS IS WRONG THING TO DO BUT WE DO IT FOR FUN.
snr, combined_snr, label_difference, filename = precision_from_repeat_visits(
    model, N_comparisons=10000, test_kwds=test_kwds)

with open(os.path.join(OUTPUT_PATH, "aspcap_restricted_trained_snr_precision.pkl"), "wb") as fp:
    pickle.dump((snr, combined_snr, label_difference, filename), fp)



raise a

# Plot theta.
fig = tc.plot.theta(model, indices=np.arange(len(label_names) + 1))
fig.set_figheight(20)
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(os.path.join(OUTPUT_PATH, "{}_theta.pdf".format(prefix)),
            dpi=300)

# Do one-to-one.
oto_labels, oto_cov, oto_meta = model.test(
    training_set_flux, training_set_ivar, **test_kwds)

fig = tc.plot.one_to_one(model, oto_labels)
fig.set_figheight(40)
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig(os.path.join(OUTPUT_PATH, "{}_oto.pdf".format(prefix)),
            dpi=300)


raise a

# Run the model on all visits and plot dispersion as a function of S/N value.
snr, combined_snr, label_difference, filename = precision_from_repeat_visits(
    model, N_comparisons=10000, test_kwds=test_kwds)

with open(os.path.join(OUTPUT_PATH, "RestrictedModel_with_aspcap_windows_precision_snr.pkl"), "wb") as fp:
    pickle.dump((snr, combined_snr, label_difference, filename), fp)


