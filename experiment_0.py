
"""
Experiment 0: Is The Cannon model used for APOGEE DR14 optimised correctly?
			  What label precision can be achieved from a polynomial (second-order)
			  model without censoring or regularisation?
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

training_set_labels, training_set_flux, training_set_ivar = setup_training_set()


OUTPUT_PATH = "experiments/0/"


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
               "TI_FE", "TIII_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]
vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)

model = tc.CannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer)

# For low accuracy:
train_kwds = dict(threads=1, op_method="l_bfgs_b", 
                  op_kwds=dict(factr=1e12, pgtol=1e-5))

test_kwds = dict()
#    initial_labels=np.percentile(model.training_set_labels, [5, 50, 95], axis=0))

model.train(**train_kwds)
model.write(os.path.join(OUTPUT_PATH, "experiment_0.model"), overwrite=True)


# Run the model on all visits and plot dispersion as a function of S/N value.
snr, combined_snr, label_difference, filename = precision_from_repeat_visits(
    model, N_comparisons=10000, test_kwds=test_kwds)

with open(os.path.join(OUTPUT_PATH, "precision_snr.pkl"), "wb") as fp:
    pickle.dump((snr, combined_snr, label_difference, filename), fp)


# Do one-to-one.
oto_labels, oto_cov, oto_meta = model.test(training_set_flux, training_set_ivar,
                                           **test_kwds)

fig = tc.plot.one_to_one(model, oto_labels)
fig.savefig(os.path.join(OUTPUT_PATH, "one_to_one.pdf"))


# Let's check if this model is trained correctly. We will re-train models of
# (TEFF, LOGG, FE_H) and one X_FE at a time, then plot the spectral derivatives
# of X_FE, X_FE * TEFF, etc and look at the difference.
for label_name in label_names[3:]:

    new_vectorizer = tc.vectorizer.PolynomialVectorizer(
        ["TEFF", "LOGG", "FE_H", label_name], 2)

    new_model = tc.CannonModel(training_set_labels, training_set_flux,
        training_set_ivar, new_vectorizer)
    new_model.train(**train_kwds)

    # Do one-to-one:

    new_oto_labels, new_oto_cov, new_oto_media = new_model.test(
        training_set_flux, training_set_ivar, **test_kwds)

    A = 2
    fig, axes = plt.subplots(A, A, figsize=(3*A, 3*A))
    axes = np.array(axes).flatten()

    for i, (ax, label_name) in enumerate(zip(axes, new_vectorizer.label_names)):

        x = training_set_labels[label_name]
        y_org = oto_labels[:, vectorizer.label_names.index(label_name)]
        y_new = new_oto_labels[:, i]

        ax.scatter(x, y_org, facecolor="b", alpha=0.5, s=1)
        ax.scatter(x, y_new, facecolor="r", alpha=0.5, s=1)
        
        limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
        limits = (min(limits), max(limits))
        ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)

        ax.text(0.05, 0.90, "{:.2f} {:.2f}".format(np.mean(y_org-x), np.std(y_org-x)),
                transform=ax.transAxes)
        
        ax.text(0.05, 0.75, "{:.2f} {:.2f}".format(np.mean(y_new-x), np.std(y_new-x)),
                transform=ax.transAxes)
        
        ax.set_title(label_name)

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_PATH, "one_to_one_4_label_{}.pdf".format(label_name)))

    fig, ax = plt.subplots(figsize=(24, 3))

    old_idx = 1 + vectorizer.label_names.index(label_name)
    new_idx = 1 + new_vectorizer.label_names.index(label_name)

    ax.plot(vacuum_wavelength, model.theta.T[old_idx], c='b',
        label="Full model")
    ax.plot(vacuum_wavelength, new_model.theta.T[new_idx], c='r',
        label="Four-label model")
    ax.set_xlim(vacuum_wavelength[0], vacuum_wavelength[-1])

    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_PATH, "theta_coefficients_{}.pdf".format(label_name)))


