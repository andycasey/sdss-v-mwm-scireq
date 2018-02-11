
# Plot precision as a function of SNR for a couple of experiments.
from plot_precision import plot_precision

default_kwds = dict(
    scatter_kwds=dict(visible=False),
    fill_between_kwds=dict(visible=False)
)
show_kwds = dict(
    fitted=True,
    scatter_kwds=dict(visible=False),
    fill_between_kwds=dict(visible=True)
)

compare_labels = ["TEFF", "LOGG", "FE_H", "C_FE", "N_FE", "O_FE", "NA_FE",
                  "MG_FE", "AL_FE", "SI_FE", "P_FE", "S_FE", "K_FE", "CA_FE",
                  "TI_FE", "V_FE", "CR_FE", "MN_FE", "CO_FE", "NI_FE"]



for show_rms in (True, False):

    suffix = "rms" if show_rms else "absdelta"

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
        ["Restricted and no cross-terms", "experiments/3", "restricted_wo_ct.model", "baseline_wo_ct_precision_snr.pkl", show_kwds],
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
        ["Restricted; no abundance cross-terms; censored", "experiments/4", "restricted_wo_ct_censored.model", "baseline_wo_ct_censored_precision_snr.pkl", show_kwds]
    ]

    fig = plot_precision(experiments, compare_labels,
                         color_offset=1, show_rms=show_rms)
    fig.savefig("experiments/precision_baseline+restricted+noct+censored_{}.pdf".format(suffix),
                dpi=300)


# Take a slice through S/N 30 +/- a bit and see what we get.



