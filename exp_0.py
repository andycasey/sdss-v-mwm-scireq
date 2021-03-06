
"""
Using DR14 training set:

[X] Train a standard Cannon model.

[X] Plot theta.

[X] Plot one-to-one.

[X] Calculate precision as a function of SNR.

[ ] Show performance on globular clusters.
"""

import numpy as np
import os
import pickle
from astropy.io import fits
from astropy.table import Table

import thecannon as tc
import thecannon.continuum

from apogee import config
from apogee.io import read_spectrum

from experiments import setup_training_set, precision_from_repeat_visits


vacuum_wavelengths, training_set_labels, training_set_flux, training_set_ivar = setup_training_set()

# ----------- #
train_kwds = dict(op_kwds=dict(factr=1e12, pgtol=1e-5))
test_kwds = dict()
OUTPUT_PATH = "experiments/0/"
# ----------- #


if not os.path.exists(OUTPUT_PATH): os.mkdir(OUTPUT_PATH)


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
               "TI_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]


# Plot the labels...
fig, ax = plt.subplots()
scat = ax.scatter(training_set_labels["TEFF"], training_set_labels["LOGG"],
    c=training_set_labels["FE_H"], s=1)
cbar = plt.colorbar(scat)
cbar.set_label("[Fe/H]")
ax.set_xlabel("TEFF")
ax.set_ylabel("LOGG")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, "training_set_teff_logg.pdf"), dpi=300)

for label_name in label_names[3:]:
    fig, ax = plt.subplots()
    ax.scatter(training_set_labels["FE_H"], training_set_labels[label_name],
        c=training_set_labels["SNR"], s=1)
    ax.set_xlabel("[FE/H]")
    ax.set_ylabel(label_name)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_PATH, "training_set_{}.pdf".format(label_name)),
        dpi=300)


vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)

model = tc.CannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer,
    dispersion=vacuum_wavelengths)

model.train(**train_kwds)
model.write(os.path.join(OUTPUT_PATH, "baseline.model"), overwrite=True)

# Plot theta.
fig = tc.plot.theta(model, indices=np.arange(len(label_names) + 1))
fig.set_figheight(20)
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(os.path.join(OUTPUT_PATH, "baseline_theta.pdf"), dpi=300)

raise a

# Do one-to-one.
oto_labels, oto_cov, oto_meta = model.test(training_set_flux, training_set_ivar,
                                           **test_kwds)

fig = tc.plot.one_to_one(model, oto_labels)
fig.savefig(os.path.join(OUTPUT_PATH, "baseline_one_to_one.pdf"))


# Run the model on all visits and plot dispersion as a function of S/N value.
snr, combined_snr, label_difference, filename = precision_from_repeat_visits(
    model, N_comparisons=1000, test_kwds=test_kwds)

with open(os.path.join(OUTPUT_PATH, "baseline_precision_snr.pkl"), "wb") as fp:
    pickle.dump((snr, combined_snr, label_difference, filename), fp)

