
# Plot precision as a function of SNR for a couple of experiments.
from plot_precision import plot_precision, plot_rms_given_snr

default_kwds = dict(
    scatter_kwds=dict(visible=False),
    fill_between_kwds=dict(visible=False)
)
show_kwds = dict(
    fitted=True,
    scatter_kwds=dict(visible=False),
    fill_between_kwds=dict(visible=True)
)
final_show_kwds = dict(fitted=True, scatter_kwds=dict(visible=False), fill_between_kwds=dict(visible=True), color_index=7)

compare_labels = ["TEFF", "LOGG", "FE_H", "C_FE", "N_FE", "O_FE", "NA_FE",
                  "MG_FE", "AL_FE", "SI_FE", "P_FE", "S_FE", "K_FE", "CA_FE",
                  "TI_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]


# Let's get a SNR dist.
import os
from apogee import config
import numpy as np
from astropy.table import Table
#t = Table.read(os.path.join(config["APOGEE_DR14_DIR"], "allStar-l31c.2.fits"))
t = Table.read("/Users/arc/research/projects/active/the-battery-stars/catalogs/allStar-l31c.2.fits")
snr = np.array(t["SNR"])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
bins = np.linspace(0, 500, 50)
num, _ = np.histogram(snr, bins)
ax.hist(snr, bins=bins, facecolor="#CCCCCC", edgecolor="#666666")
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
value = bins[np.argmax(num)] + np.diff(bins)[0]/2.0
ax.axvline(value, c="k", linestyle=":", lw=1)
ax.set_xticks([value])
ax.set_xticklabels(["SNR"])
fig.tight_layout()
fig.savefig("snr.pdf", dpi=300)


"""
for show_rms in (True, False):

    suffix = "rms" if show_rms else "absdelta"

    experiments = [
        # Label, experiment, model basename, result basename, kwds
        ["DR14", "experiments/0", "baseline.model", "baseline_precision_snr.pkl", show_kwds],
        #["Baseline", "experiments/1", "baseline.model", "baseline_precision_snr.pkl", 
        #    dict(fitted=True, scatter_kwds=dict(visible=False), fill_between_kwds=dict(visible=True), color_index=1)],
    ]

    for snr_value in (30, 50, 100, 200):
        fig  = plot_rms_given_snr(snr * snr_value/value, experiments, compare_labels, line_value=0.20)
        fig.savefig("experiments/snr_sigma_{}.pdf".format(snr_value), dpi=300)

    experiments = [
        # Label, experiment, model basename, result basename, kwds
        ["DR14", "experiments/0", "baseline.model", "baseline_precision_snr.pkl", show_kwds],
        ["Restricted; no abundance cross-terms; censored", "experiments/4", "restricted_wo_ct_censored.model", "baseline_wo_ct_censored_precision_snr.pkl", final_show_kwds]
    ]

    for snr_value in (30, 50, 100, 200):
        fig  = plot_rms_given_snr(snr * snr_value/value, experiments, compare_labels, line_value=0.20)
        fig.savefig("experiments/snr_sigma_all_{}.pdf".format(snr_value), dpi=300)


    raise a

    fig = plot_precision(experiments, compare_labels, show_rms=show_rms)
    fig.savefig("experiments/precision_baseline+censored_{}.pdf".format(suffix),
                dpi=300)
"""

for show_rms in (True, False):

    suffix = "rms" if show_rms else "absdelta"
    experiments = [
        # Label, experiment, model basename, result basename, kwds
        ["DR14", "experiments/0", "baseline.model", "baseline_precision_snr.pkl", show_kwds],
    ]
    fig = plot_precision(experiments, compare_labels, show_rms=show_rms)
    fig.savefig("experiments/precision_dr14_{}.pdf".format(suffix),
                dpi=300)


    raise a

    experiments = [
        # Label, experiment, model basename, result basename, kwds
        ["DR14", "experiments/0", "baseline.model", "baseline_precision_snr.pkl", show_kwds],
        ["Baseline", "experiments/1", "baseline.model", "baseline_precision_snr.pkl", show_kwds],
    ]
    fig = plot_precision(experiments, compare_labels, show_rms=show_rms)
    fig.savefig("experiments/precision_dr14+baseline_{}.pdf".format(suffix),
                dpi=300)


    experiments = [
        # Label, experiment, model basename, result basename, kwds
        ["Baseline", "experiments/1", "baseline.model", "baseline_precision_snr.pkl", show_kwds],
        ["Restricted", "experiments/2", "restricted.model", "restricted_precision_snr.pkl", show_kwds]
    ]

    fig = plot_precision(experiments, compare_labels,
                         color_offset=1, show_rms=show_rms)
    fig.savefig("experiments/precision_baseline+restricted_{}.pdf".format(suffix),
                dpi=300)

    experiments = [
        # Label, experiment, model basename, result basename, kwds
        ["Baseline", "experiments/1", "baseline.model", "baseline_precision_snr.pkl", show_kwds],
        ["Restricted", "experiments/2", "restricted.model", "restricted_precision_snr.pkl", show_kwds],
        ["Restricted; no abundance cross-terms", "experiments/3", "restricted_wo_ct.model", "baseline_wo_ct_precision_snr.pkl", show_kwds],
    ]
    fig = plot_precision(experiments, compare_labels,   
                         color_offset=1, show_rms=show_rms)
    fig.savefig("experiments/precision_baseline+restricted+noct_{}.pdf".format(
        suffix), dpi=300)

    experiments = [
        # Label, experiment, model basename, result basename, kwds
        ["Baseline", "experiments/1", "baseline.model", "baseline_precision_snr.pkl", show_kwds],
        ["Restricted", "experiments/2", "restricted.model", "restricted_precision_snr.pkl", show_kwds],
        ["Restricted; no abundance cross-terms", "experiments/3", "restricted_wo_ct.model", "baseline_wo_ct_precision_snr.pkl", show_kwds],
        ["Restricted; no abundance cross-terms; censored", "experiments/4", "restricted_wo_ct_censored.model", "baseline_wo_ct_censored_precision_snr.pkl", final_show_kwds]
    ]

    fig = plot_precision(experiments, compare_labels,
                         color_offset=1, show_rms=show_rms)
    fig.savefig("experiments/precision_baseline+restricted+noct+censored_{}.pdf".format(suffix),
                dpi=300)


# Take a slice through S/N 30 +/- a bit and see what we get.



