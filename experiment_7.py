
"""
Experiment 7:   Construct a training set for APOGEE DR14.
                Use a RestrictedCannonModel without cross-terms (and use
                regularisation?).
                Examine how we perform on globular clusters compared to ASPCAP.
"""

import numpy as np
import os
import pickle
from astropy.table import Table


import thecannon as tc
import thecannon.restricted

import apogee

from apogee import config
from experiments import training_set_data


OVERWRITE = True
OUTPUT_PATH = "experiments/7"

test_kwds = dict()
train_kwds = dict(op_kwds=dict(factr=1e12, pgtol=1e-5))

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

stars = Table.read(
    os.path.join(config["APOGEE_DR14_DIR"], "allStar-l31c.2.fits"))

validation_set = (stars["SNR"] > 200) \
                * (stars["ASPCAPFLAG"] == 0) \
                * (stars["ASPCAP_CHI2"] < 50) \
                * ~apogee.bitmasks.has_any_starflag(stars["STARFLAG"], [
                    "PERSIST_HIGH", "PERSIST_JUMP_POS", "PERSIST_JUMP_NEG",
                    "SUSPECT_BROAD_LINES", "SUSPECT_RV_COMBINATION",
                    "VERY_BRIGHT_NEIGHBOUR"
                ])


m, c = (2.6/1100, -7.70)
logg_criteria = stars["TEFF"] * m + c
validation_set *= (stars["LOGG"] <= logg_criteria)

# We want stars with abundances.....
# We can get the other abundances, but these are the ones we want to verify
# our model with.
label_names = ["TEFF", "LOGG", "FE_H", "C_FE", "N_FE", "O_FE", "NA_FE", 
               "MG_FE", "AL_FE", "SI_FE", "K_FE", "CA_FE", "TI_FE",
               "CR_FE", "MN_FE", "NI_FE"]

original_validation_set = validation_set.copy()
for label_name in label_names:
    
    removed = (stars[label_name][original_validation_set] < -5000)
    print("We removed {} good stars due to missing {}".format(
        sum(removed), label_name))

    validation_set *= (stars[label_name] > -5000)
    

# Let's create a balanced training set of about 1,000 stars.
N_bins = 10

uln = ["TEFF", "LOGG", "FE_H", "O_FE", "NA_FE", "MG_FE", "AL_FE"]
unbalanced_data = np.array([stars[ln][validation_set] for ln in uln]).T

H, bins = np.histogramdd(unbalanced_data, N_bins)
indices = np.array([np.digitize(c, b) for c, b in zip(unbalanced_data.T, bins)]) - 1

assert np.sum(np.all(indices.T == np.array(np.where(H == H.max())).flatten(), axis=1)) == H.max()

"""
Here we will just take 1 star from every multidimensional bin that has a count
of at least 1. That gives us some 5000 stars.
If you want a larger sample size then you may need to over-sample some bins,
or accept that you will have a *slightly* "unbalanced" data set (but far more
balanced than what it was to begin with).
"""

np.random.seed(42)

use_bins = np.array(np.where(H >= 1)).T
in_training_set = np.zeros(len(unbalanced_data), dtype=bool)

for i, bin_indices in enumerate(use_bins):

    star_indices = np.where(np.all(indices.T == bin_indices, axis=1))[0]
    print(i, star_indices.size)

    if 1 > star_indices.size:
        print("What the fuck?")
        continue

    in_training_set[np.random.choice(star_indices, size=1)] = True


star_indices = np.where(validation_set)[0][in_training_set]
training_set = np.zeros(len(stars), dtype=bool)
training_set[star_indices] = True


fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].scatter(stars["TEFF"][validation_set], stars["LOGG"][validation_set], 
    c=stars["FE_H"][validation_set], s=1)
axes[1, 0].scatter(stars["TEFF"][training_set], stars["LOGG"][training_set], 
    c=stars["FE_H"][training_set], s=1)

for ax in axes[:, 0]:
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_xlabel("TEFF")
    ax.set_ylabel("LOGG")


axes[0, 1].scatter(stars["NA_FE"][validation_set], stars["O_FE"][validation_set],
    c=stars["FE_H"][validation_set], s=1)
axes[1, 1].scatter(stars["NA_FE"][training_set], stars["O_FE"][training_set],
    c=stars["FE_H"][training_set], s=1)


for ax in axes[:, 1]:
    ax.set_xlabel("[Na/Fe]")
    ax.set_ylabel("[O/Fe]")

fig.tight_layout()

fig.savefig(os.path.join(OUTPUT_PATH, "validation_training_sets.pdf"), dpi=300)


# Construct training set.
training_set_labels = stars[training_set]
vacuum_wavelengths, training_set_flux, training_set_ivar \
    = training_set_data(training_set_labels)



fig, ax = plt.subplots()
for i in range(100):
    ax.plot(vacuum_wavelengths, training_set_flux[i], c='k', alpha=0.01)



# Train a RestrictedCannonModel
vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order=2)
theta_bounds = dict([(l, (None, 0)) for l in label_names if l.endswith("_FE")])

model = tc.restricted.RestrictedCannonModel(
    training_set_labels, training_set_flux, training_set_ivar, vectorizer, 
    theta_bounds=theta_bounds)

model.train(**train_kwds)
model.write(os.path.join(OUTPUT_PATH, "restricted.model"), overwrite=OVERWRITE)


# Plot theta.
fig = tc.plot.theta(model, indices=np.arange(len(label_names) + 1))
fig.set_figheight(20)
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(os.path.join(OUTPUT_PATH, "restricted_theta.pdf"), dpi=300)

raise a


# Do one-to-one.
oto_labels, oto_cov, oto_meta = model.test(
    training_set_flux, training_set_ivar, **test_kwds)

fig = tc.plot.one_to_one(model, oto_labels)
fig.set_figheight(40)
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig(os.path.join(OUTPUT_PATH, "restricted_one_to_one.pdf"), dpi=300)


# Run test-step on globular cluster star spectra.



# MODEL: RestrictedCannonModel without cross-terms (and regularisation?)





# SELECT A TRAINING SET AND KNOW HOW MANY GC STARS ARE IN IT.

# TRAIN A RESTRICTEDCANNONMODEL (WITH REGULARISATION)

# 