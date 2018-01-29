
"""
Experiment 1: Can we make restrict the amount of information in abundance
              correlations by limiting the abundance coefficients to be
              negative, and removing abundance cross-terms from the vectorizer?
"""

import numpy as np
import os

import thecannon as tc
import thecannon.restricted

from experiments import setup_training_set, precision_from_repeat_visits

# ---------------- #

OUTPUT_PATH = "experiments/1"

test_kwds = dict()
train_kwds = dict(op_method="l_bfgs_b", op_kwds=dict(factr=1e12, pgtol=1e-5))

label_names = ["TEFF", "LOGG", "FE_H", "C_FE", "CI_FE", "N_FE", "O_FE", "NA_FE", 
               "MG_FE", "AL_FE", "SI_FE", "P_FE", "S_FE", "K_FE", "CA_FE", 
               "TI_FE", "TIII_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]

# ---------------- #

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

training_set_labels, training_set_flux, training_set_ivar = setup_training_set()


"""
Train a new model using bounds on the abundance label coefficients.
"""
vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)
theta_bounds = dict([(l, (None, 0)) for l in label_names if l.endswith("_FE")])


model_with_bounds = tc.restricted.RestrictedCannonModel(training_set_labels, 
    training_set_flux, training_set_ivar, vectorizer, theta_bounds=theta_bounds)

model_with_bounds.train(**train_kwds)
model_with_bounds.write(os.path.join(OUTPUT_PATH, "model_with_bounds.model"))


"""
# Plot theta.
fig = tc.plot.theta(model_with_bounds, indices=np.arange(len(label_names) + 1))
fig.set_figheight(20)
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(os.path.join(OUTPUT_PATH, "restricted_theta.pdf"), dpi=300)


# Do one-to-one.
oto_labels, oto_cov, oto_meta = model_with_bounds.test(
    training_set_flux, training_set_ivar, **test_kwds)

fig = tc.plot.one_to_one(model_with_bounds, oto_labels)
fig.set_figheight(40)
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig(os.path.join(OUTPUT_PATH, "restricted_one_to_one.pdf"), dpi=300)

"""

# Run the model on all visits and plot dispersion as a function of S/N value.
wb_snr, wb_combined_snr, wb_label_difference, wb_filename = \
    precision_from_repeat_visits(model_with_bounds, test_kwds=test_kwds)


"""
Train a new model without abundance cross-terms.
"""

# Restrict the vectorizer to only use cross-terms with TEFF and LOGG:
new_terms = \
    [t for t in vectorizer.get_human_readable_label_vector().split(" + ")[1:]
        if "*" not in t or ("TEFF" in t.split("*") or "LOGG" in t.split("*"))]
vectorizer_wo_ct = tc.vectorizer.PolynomialVectorizer(terms=new_terms)

model_wo_ct = tc.CannonModel(training_set_labels, training_set_flux,
    training_set_ivar, vectorizer_wo_ct)
model_wo_ct.train(**train_kwds)
model_wo_ct.write(os.path.join(OUTPUT_PATH, "model_wo_ct.model"))

"""
# Plot all theta.
fig = tc.plot.theta(model_wo_ct)
fig.set_figheight(20)
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(os.path.join(OUTPUT_PATH, "wo_ct_theta.pdf"), dpi=300)

# Do one-to-one.
oto_wo_ct_labels, oto_wo_ct_cov, oto_wo_ct_meta = model_wo_ct.test(
    training_set_flux, training_set_ivar, **test_kwds)

fig = tc.plot.one_to_one(model_wo_ct, oto_wo_ct_labels)
fig.set_figheight(40)
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig(os.path.join(OUTPUT_PATH, "wo_ct_one_to_one.pdf"), dpi=300)
"""

woct_snr, woct_combined_snr, woct_label_difference, woct_filename = \
    precision_from_repeat_visits(model_wo_ct, test_kwds=test_kwds)


"""
Train a new model using bounds on the abundance label coefficients,
*and* without abundance cross-terms.
"""


model_wb_and_wo_ct = tc.restricted.RestrictedCannonModel(training_set_labels,
    training_set_flux, training_set_ivar, vectorizer_wo_ct, 
    theta_bounds=theta_bounds)
model_wb_and_wo_ct.train(**train_kwds)
model_wb_and_wo_ct.write(os.path.join(OUTPUT_PATH, "model_wb_and_wo_ct.model"))

"""
# Plot all theta.
fig = tc.plot.theta(model_wb_and_wo_ct, indices=np.arange(len(label_names) + 1))
fig.set_figheight(20)
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(os.path.join(OUTPUT_PATH, "wb_and_wo_ct_theta.pdf"), dpi=300)

# Do one-to-one.
oto_wb_wo_ct_labels, oto_wb_wo_ct_cov, oto_wb_wo_ct_meta = model_wb_and_wo_ct.test(
    training_set_flux, training_set_ivar, **test_kwds)

fig = tc.plot.one_to_one(model_wb_and_wo_ct, oto_wb_wo_ct_labels)
fig.set_figheight(40)
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig(os.path.join(OUTPUT_PATH, "wb_and_wo_ct_one_to_one.pdf"), dpi=300)
"""

wb_wo_ct_snr, wb_wo_ct_combined_snr, wb_wo_ct_label_difference, wb_wo_ct_filename \
    = precision_from_repeat_visits(model_wb_and_wo_ct, test_kwds=test_kwds)

