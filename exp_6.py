
"""
Using our balanced training set:

[X] Train a restricted Cannon model without cross-terms, and censor the windows
    based on the ASPCAP windows. Note that we just use the windows without
    weights because I haven't figured out how to include them yet.
"""


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import thecannon as tc
import thecannon.restricted

from apogee import config

from experiments import get_balanced_training_set, precision_from_repeat_calibration_visits
from utils import atomic_number
label_names = ["TEFF", "LOGG", "FE_H", "C_FE", "N_FE", "O_FE", "NA_FE",
               "MG_FE", "AL_FE", "SI_FE", "P_FE", "S_FE", "K_FE", "CA_FE",
               "TI_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]
label_names_for_balancing = ["TEFF", "LOGG", "FE_H", "NA_FE", "O_FE", "MG_FE",
                             "AL_FE"]

vacuum_wavelengths, training_set_labels, training_set_flux, training_set_ivar \
    = get_balanced_training_set(label_names, label_names_for_balancing)


# ----------- #
train_kwds = dict()
test_kwds = dict()
OUTPUT_PATH = "experiments/6/"
# ----------- #

if not os.path.exists(OUTPUT_PATH): os.mkdir(OUTPUT_PATH)

censors = {}
weights = {}
for label_name in label_names:
    if "_" not in label_name: continue

    element = label_name.split("_")[0]
    Z = atomic_number(element)

    path = os.path.join(
        config["DATA_DIR"],
        "dr14/apogee/windows",
        "{}_DR13_Window_Ranges_VAC.csv".format(element.title()))

    data = np.loadtxt(path, delimiter=",", usecols=(0, 1))

    censors[label_name] = np.zeros(vacuum_wavelengths.size, dtype=bool)
    weights[label_name] = np.zeros(vacuum_wavelengths.size, dtype=float)

    indices = np.searchsorted(vacuum_wavelengths, data.T[0])
    censors[label_name][indices] = True
    weights[label_name][indices] = data.T[1]



model_path = os.path.join(OUTPUT_PATH, "aspcap_censored.model")
if os.path.exists(model_path):
    model = tc.CannonModel.read(model_path)

else:

    vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)

    """
    # Restrict the vectorizer to only use cross-terms with TEFF and LOGG:
    new_terms = \
        [t for t in vectorizer.get_human_readable_label_vector().split(" + ")[1:]
            if "*" not in t or ("TEFF" in t.split("*") or "LOGG" in t.split("*"))]
    vectorizer = tc.vectorizer.PolynomialVectorizer(terms=new_terms)


    model = tc.restricted.RestrictedCannonModel(
        training_set_labels, training_set_flux, training_set_ivar, vectorizer,
        dispersion=vacuum_wavelengths, censors=censors,
        theta_bounds=dict([(ln, (None, 0)) for ln in label_names if ln.endswith("_FE")]))
    """



    model = tc.CannonModel(training_set_labels, training_set_flux, training_set_ivar,
        vectorizer, dispersion=vacuum_wavelengths, censors=censors)
    model.train(**train_kwds)

    # Some pixels had issues. Re-train using current coefficients as initial values.
    model.train(op_method="powell")

    model.write(model_path,overwrite=True)


# Plot theta.
fig = tc.plot.theta(model, indices=np.arange(len(label_names) + 1))
fig.set_figheight(20)
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(
    os.path.join(OUTPUT_PATH, "aspcap_censored_theta.pdf"),
    dpi=300)


# Plot the weights compared to the spectral derivatives?????
fig, axes = plt.subplots(4, 5, figsize=(16, 20))
axes = np.array(axes).flatten()
for ax, label_name in zip(axes, label_names):

    ax.text(0.9, 0.9, label_name, 
        transform=ax.transAxes, horizontalalignment="right")
    
    if "_" not in label_name: continue

    theta_indices = model.theta.T[1 + model.vectorizer.label_names.index(label_name)]
    pixel_indices = weights[label_name] > 0

    ax.scatter(
        weights[label_name][pixel_indices],
        theta_indices[pixel_indices],
        s=1, c="#000000")
    ax.set_xlabel(r"weight")
    ax.set_ylabel(r"$\theta$")

fig.tight_layout()
fig.savefig(
    os.path.join(OUTPUT_PATH, "aspcap_censored_weights-vs-theta.pdf"),
    dpi=300)


#visit_snr, combined_snr, visit_snr_labels, combined_snr_labels, apogee_ids \
results_path = os.path.join(OUTPUT_PATH, "aspcap_censored_6.pkl")
if os.path.exists(results_path):
    with open(results_path, "rb") as fp:
        results = pickle.load(fp)

else:
    results = precision_from_repeat_calibration_visits(model, N_comparisons=10000)
    with open(results_path, "wb") as fp:
        pickle.dump(results, fp)


raise a



model = tc.restricted.CannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer, 
    dispersion=vacuum_wavelengths, censors=censors)

model.train(**train_kwds)
model.write(
    os.path.join(OUTPUT_PATH, "aspcap_censored_restricted.model"),
    overwrite=True)



# Plot the weights compared to the spectral derivatives?????
fig, axes = plt.subplots(4, 5, figsize=(16, 20))
axes = np.array(axes).flatten()
for ax, label_name in zip(axes, label_names):
    if "_" not in label_name: continue

    theta_indices = model.theta.T[1 + model.vectorizer.label_names.index(label_name)]
    pixel_indices = weights[label_name] > 0

    ax.scatter(
        weights[label_name][pixel_indices],
        theta_indices[pixel_indices],
        s=1, c="#000000")
    ax.text(0.9, 0.9, label_name, transform=ax.transAxes)
    ax.set_xlabel(r"weight")
    ax.set_ylabel(r"$\theta$")

fig.tight_layout()
fig.savefig(
    os.path.join(OUTPUT_PATH, "aspcap_censored_restricted_weights-vs-theta.pdf"),
    dpi=300)






raise a



# Do one-to-one.
oto_labels, oto_cov, oto_meta = model.test(training_set_flux, training_set_ivar,
                                           **test_kwds)

fig = tc.plot.one_to_one(model, oto_labels)
fig.savefig(
    os.path.join(OUTPUT_PATH, "aspcap_censored_restricted_noct_oto.pdf"),
    dpi=300)


# Run the model on individual visits from the calibration set.
#visit_snr, combined_snr, visit_snr_labels, combined_snr_labels, apogee_ids \
results = precision_from_repeat_calibration_visits(model, N_comparisons=10000)
with open(os.path.join(OUTPUT_PATH, "calibration_precision_6.pkl"), "wb") as fp:
    pickle.dump(results, fp)

