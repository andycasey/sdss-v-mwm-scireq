"""
exp_4_53.py

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

from experiments import get_globular_cluster_candidates



# ----------- #
OUTPUT_PATH = "experiments/4/"
# ----------- #

if not os.path.exists(OUTPUT_PATH): os.mkdir(OUTPUT_PATH)


model_path = os.path.join(OUTPUT_PATH, "restricted_wo_ct_censored.model")
model = tc.CannonModel.read(model_path)

train_kwds = dict(op_kwds=dict(factr=1e12, pgtol=1e-5))
test_kwds = dict(initial_labels=\
    np.percentile(model.training_set_labels, np.linspace(0, 100, 10), axis=0))



results_path = os.path.join(OUTPUT_PATH, "gc_results.pkl")

if not os.path.exists(results_path):
    vacuum_wavlengths, gc_candidates, gc_candidate_flux, gc_candidate_ivar \
        = get_globular_cluster_candidates(("M92", "M53", "M15", "M13"))

    test_labels, test_cov, test_meta = model.test(
        gc_candidate_flux, gc_candidate_ivar, **test_kwds)

    with open(results_path, "wb") as fp:
        pickle.dump((
            vacuum_wavlengths, gc_candidate, gc_candidate_flux, 
            gc_candidate_flux, test_labels, test_cov, test_meta), fp, -1)


else:
    with open(results_path, "rb") as fp:
        vacuum_wavlengths, gc_candidate_flux, gc_candidate_flux, \
            gc_candidate_flux, test_labels, test_cov, test_meta = pickle.load(fp)



# Plot membership/
fig, ax = plt.subplots()


feh = test_labels.T[model.vectorizer.label_names.index("FE_H")]
ax.scatter(
    gc_candidates["VHELIO_AVG"], 
    feh,
    c="#CCCCCC")

membership = (gc_candidates["VHELIO_AVG"] >= -140) \
           * (gc_candidates["VHELIO_AVG"] <= -100) \
           * (feh < -1.35) * (feh > -1.55)
ax.scatter(
    gc_candidates["VHELIO_AVG"][membership],
    feh[membership],
    facecolor="r")

ax.set_xlabel("VHELIO_AVG")
ax.set_ylabel("FE_H (The Cannon)")
fig.tight_layout()


def latex_label(label_name):
    return label_name



# Abundances.
def plot_globular_cluster_abundances(model, gc_candidates, test_labels,
    membership):

    aspcap_candidate_kwds = dict(s=1, c="#666666")
    aspcap_membership_kwds = dict(s=2, c="r")

    cannon_candidate_kwds = aspcap_candidate_kwds.copy()
    cannon_membership_kwds = dict(s=2, c="g")

    label_comparisons = [
        ("C_FE", "N_FE"),
        ("O_FE", "NA_FE"),
        ("MG_FE", "AL_FE"),
        ("CA_FE", "S_FE"),
    ]

    fig, axes = plt.subplots(3, 4)
    axes = np.array(axes).flatten()

    for i, (xlabel, ylabel) in enumerate(label_comparisons):

        # ASPCAP first.
        ax = axes[2*i]

        finite = (gc_candidates[xlabel] > -5000) \
               * (gc_candidates[ylabel] > -5000)
        ax.scatter(
            gc_candidates[xlabel][finite * ~membership],
            gc_candidates[ylabel][finite * ~membership],
            **aspcap_candidate_kwds)

        ax.scatter(
            gc_candidates[xlabel][finite * membership],
            gc_candidates[ylabel][finite * membership],
            **aspcap_membership_kwds)

        if ax.is_first_row():
            ax.set_title("ASPCAP")

        # The Cannon.
        ax = axes[2*i + 1]

        xidx = model.vectorizer.label_names.index(xlabel)
        yidx = model.vectorizer.label_names.index(ylabel)

        ax.scatter(
            test_labels.T[xidx][~membership],
            test_labels.T[yidx][~membership],
            **cannon_candidate_kwds)
        ax.scatter(
            test_labels.T[xidx][membership],
            test_labels.T[yidx][membership],
            **cannon_membership_kwds)

        if ax.is_first_row():
            ax.set_title("The Cannon")

        xlims = np.array([ax.get_xlim() for ax in axes[2*i:2*i + 2]])
        ylims = np.array([ax.get_ylim() for ax in axes[2*i:2*i + 2]])


        for ax in axes[2*i:2*i + 2]:
            ax.set_xlabel(latex_label(xlabel))
            ax.set_ylabel(latex_label(ylabel))

            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))

            ax.set_xlim(np.min(xlims), np.max(xlims))
            ax.set_ylim(np.min(ylims), np.max(ylims))

            #ax.set_xlim(axes[2*i + 1].get_xlim())
            #ax.set_ylim(axes[2*i + 1].get_ylim())



    return fig







