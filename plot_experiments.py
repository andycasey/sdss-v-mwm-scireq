

import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.cm import Dark2 as colormap
from scipy import ndimage

import thecannon as tc



# Plot precision as a function of SNR for a couple of experiments.

default_kwds = dict(
    scatter_kwds=dict(visible=False),
    fill_between_kwds=dict(visible=False)
)
show_kwds = dict(
    scatter_kwds=dict(visible=False),
    fill_between_kwds=dict(visible=True)
)

experiments = [
    ["0", "experiments/0/experiment_0.model", "experiments/0/precision_wrt_snr.pkl", show_kwds],
    ["1a", "experiments/1/model_with_bounds.model", "experiments/1/precision_snr_wb.pkl", default_kwds],
    ["1b", "experiments/1/model_wo_ct.model", "experiments/1/precision_snr_woct.pkl", default_kwds],
    ["2", "experiments/2/aspcap_censored.model", "experiments/2/aspcap_censored_precision_snr.pkl", show_kwds],
    ["3", "experiments/3/RestrictedModel_with_aspcap_windows.model", "experiments/3/RestrictedModel_with_aspcap_windows_precision_snr.pkl", default_kwds],
]
    

compare_labels = ["TEFF", "LOGG", "FE_H"]
compare_labels = None

if compare_labels is None:
    # Find all labels from those models

    all_labels = []
    for model_name, model_path, _, __ in experiments:
        model = tc.CannonModel.read(model_path)

        for label in model.vectorizer.label_names:
            if label not in all_labels:
                all_labels.append(label)

    compare_labels = all_labels


def equal_histogram_bins(data, N_bins):
    N = len(data)
    return np.interp(np.linspace(0, N, N_bins + 1), np.arange(N), np.sort(data))


square = True
L = len(compare_labels)
if square:
    K = int(np.ceil(np.sqrt(L)))
    fig, axes = plt.subplots(K, K)

else:
    fig, axes = plt.subplots(L)

axes = np.atleast_1d(axes).flatten()




for i, (ax, label_name) in enumerate(zip(axes, compare_labels)):


    for j, (model_name, model_path, precision_path, kwds) in enumerate(experiments):

        model = tc.CannonModel.read(model_path)

        with open(precision_path, "rb") as fp:
            snr, combined_snr, label_differences, filenames = pickle.load(fp)


        try:
            label_index = model.vectorizer.label_names.index(label_name)

        except ValueError:
            print("Could not find {} in model stored at {}".format(
                label_name, model_path))
            continue

        y = np.abs(label_differences[:, label_index])

        color = colormap(j)
        
        scatter_kwds = dict(s=1, alpha=0.1, c=color)
        scatter_kwds.update(kwds.get("scatter_kwds", {}))

        ax.scatter(snr, y, **scatter_kwds)

        # Show median and percentiles in each.
        N_bins = kwds.get("N_bins", int(len(snr) / 100))
        percentiles = kwds.get("percentiles", [16, 50, 84])

        bins = equal_histogram_bins(snr, N_bins)
        centers = bins[:-1] + np.diff(bins)/2.0
        indices = np.digitize(snr, bins)

        running_percentiles = np.nan * np.ones((N_bins, 3))
        for k in range(1, 1 + N_bins):
            if k not in indices: continue
            running_percentiles[k - 1] = np.percentile(y[indices == k], percentiles)


        plot_kwds = dict(lw=2, c=color, label=model_name)
        plot_kwds.update(kwds.get("plot_kwds", {}))

        fill_between_kwds = dict(alpha=0.5, facecolor=color, edgecolor=None)
        fill_between_kwds.update(kwds.get("fill_between_kwds", {}))

        ax.plot(centers, running_percentiles.T[1], **plot_kwds)
        ax.fill_between(
            centers, running_percentiles.T[0], running_percentiles.T[2],
            **fill_between_kwds)



for ax, label_name in zip(axes, compare_labels):
    
    ax.set_xlim(0, 100)

    if ax.is_last_row():
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.set_xlabel(r"S/N")

    else:
        ax.set_xticklabels([])

    ylim_upper = 250 if label_name.upper() == "TEFF" else 1
    ax.set_ylim(0, ylim_upper)

    ax.yaxis.set_major_locator(MaxNLocator(4))

    ax.set_ylabel(label_name)


for ax in axes[len(compare_labels):]:
    ax.set_visible(False)

fig.tight_layout()
fig.subplots_adjust(wspace=0.05, hspace=0.05)

axes[0].legend()

raise a
