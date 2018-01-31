
"""
Experiment 4: Use RestrictedCannonModel and regularization.
"""

import numpy as np
import os
import pickle

import thecannon as tc
import thecannon.restricted

from apogee.aspcap import element_window
from experiments import setup_training_set, precision_from_repeat_visits

# ---------------- #

OUTPUT_PATH = "experiments/4"

test_kwds = dict()
train_kwds = dict(op_method="l_bfgs_b", op_kwds=dict(factr=1e12, pgtol=1e-5))

label_names = ["TEFF", "LOGG", "FE_H", "C_FE", "N_FE", "O_FE", "NA_FE", 
               "MG_FE", "AL_FE", "SI_FE", "P_FE", "S_FE", "K_FE", "CA_FE", 
               "TI_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]

# ---------------- #

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


vacuum_wavelengths, training_set_labels, training_set_flux, training_set_ivar \
    = setup_training_set(full_output=True)

vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)

for regularization in (0, 1000):

    model = tc.restricted.RestrictedCannonModel(
        training_set_labels, training_set_flux, training_set_ivar, vectorizer,
        regularization=regularization,
        theta_bounds=dict([(ln, (None, 0)) for ln in label_names if ln.endswith("_FE")]))
    model.train(**train_kwds)

    model.write(os.path.join(OUTPUT_PATH,
        "RestrictedModel_regularised_{:.2e}.model".format(regularization)),
        overwrite=True)


    # Plot theta.
    fig = tc.plot.theta(model, indices=np.arange(len(label_names) + 1))
    fig.set_figheight(20)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(os.path.join(OUTPUT_PATH,
        "RestrictedModel_regularised_{:.2e}_theta.pdf".format(regularization)),
        dpi=300)

    # Do one-to-one.
    oto_labels, oto_cov, oto_meta = model.test(
        training_set_flux, training_set_ivar, **test_kwds)

    fig = tc.plot.one_to_one(model, oto_labels)
    fig.set_figheight(40)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(os.path.join(OUTPUT_PATH, 
        "RestrictedModel_reguliarsed_one_to_one_{:.2e}.pdf".format(regularization)),
        dpi=300)


raise a

# Run the model on all visits and plot dispersion as a function of S/N value.
snr, combined_snr, label_difference, filename = precision_from_repeat_visits(
    model, N_comparisons=10000, test_kwds=test_kwds)

with open(os.path.join(OUTPUT_PATH, "RestrictedModel_with_aspcap_windows_precision_snr.pkl"), "wb") as fp:
    pickle.dump((snr, combined_snr, label_difference, filename), fp)


