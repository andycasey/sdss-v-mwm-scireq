

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.cm import Dark2 as colormap
from matplotlib.ticker import MaxNLocator

import thecannon as tc



def equal_histogram_bins(data, N_bins):
    N = len(data)
    return np.interp(np.linspace(0, N, N_bins + 1), np.arange(N), np.sort(data))

def latex_label(label_name):

    labels = {
        "TEFF": r"$T_{\rm eff}$",
        "LOGG": r"$\log{g}$",
        "FE_H": r"$[{\rm Fe}/{\rm H}]$"
    }
    default_label = r"$[{{\rm {0}}}/{{\rm Fe}}]$".format(label_name.split("_")[0].title())
    return labels.get(label_name, default_label)


def plot_precision(experiments, compare_labels, square=True, xlim_upper=100,
    ylim_uppers=None, color_offset=0, show_rms=True, **kwargs):

    if ylim_uppers is None:
        ylim_uppers = {}

    L = len(compare_labels)
    if square:
        K = int(np.ceil(np.sqrt(L)))
        M = (K - 1) if (K - 1) * K <= L else K
        fig, axes = plt.subplots(K, M, figsize=(1.75*K, 1.75*M))

    else:
        fig, axes = plt.subplots(L)

    axes = np.atleast_1d(axes).flatten()


    handles = []
    labels = []

    for i, (ax, label_name) in enumerate(zip(axes, compare_labels)):


        for j, (model_name, experiment, model_basename, result_basename, kwds) in enumerate(experiments):

            model_path = os.path.join(experiment, model_basename)
            results_path = os.path.join(experiment, result_basename)

            model = tc.CannonModel.read(model_path)
            with open(results_path, "rb") as fp:
                snr, combined_snr, label_differences, filenames = pickle.load(fp)

            try:
                label_index = model.vectorizer.label_names.index(label_name)

            except ValueError:
                print("Could not find {} in model stored at {}".format(
                    label_name, model_path))
                continue

            if experiment == "experiments/3" and "CI_FE" in model.vectorizer.label_names \
            and model.vectorizer.label_names.index("CI_FE") <= label_index:
                print("WARNING HAVING TO CORRECT HACK")
                label_index -= 1
            y = np.abs(label_differences[:, label_index])

            color = colormap(kwds.pop("color_index", j + color_offset))
            
            scatter_kwds = dict(s=1, alpha=0.1, c=color)
            scatter_kwds.update(kwds.get("scatter_kwds", {}))

            ax.scatter(snr, y, **scatter_kwds)

            # Show median and percentiles in each.
            N_bins = kwds.get("N_bins", int(len(snr) / 50))
            percentiles = kwds.get("percentiles", [16, 50, 84])

            bins = equal_histogram_bins(snr, N_bins)
            centers = bins[:-1] + np.diff(bins)/2.0
            indices = np.digitize(snr, bins)

            if not show_rms:

                running_percentiles = np.nan * np.ones((N_bins, 3))
                for k in range(1, 1 + N_bins):
                    if k not in indices: continue
                    running_percentiles[k - 1] = np.percentile(y[indices == k], percentiles)

                # Just fit a polynomial.
                fitted_percentiles = np.zeros_like(running_percentiles)
                for k, each in enumerate(running_percentiles.T):
                    p = np.polyfit(centers, each, 5)
                    fitted_percentiles[:, k] = np.polyval(p, centers)

                show_percentiles = fitted_percentiles if kwds.get("fitted", True) \
                                                      else running_percentiles

                fill_between_kwds = dict(alpha=0.5, facecolor=color, edgecolor=None)
                fill_between_kwds.update(kwds.get("fill_between_kwds", {}))

                show_y = show_percentiles.T[1]

                ax.fill_between(
                    centers, show_percentiles.T[0], show_percentiles.T[2],
                    **fill_between_kwds)

            else:
                rms = np.nan * np.ones(N_bins)
                for k in range(N_bins):
                    if k not in indices: continue
                    N = np.sum(indices == k)
                    rms[k] = np.sqrt(np.sum((y[indices == k])**2)/N)

                if kwds.get("fitted", True):
                    finite = np.isfinite(rms)
                    p = np.polyfit(centers[finite], rms[finite], 5)
                    show_y = np.polyval(p, centers)
                else:
                    show_y = rms

                scat_kwds = dict(c=color, s=5)
                ax.scatter(centers, show_y, **scat_kwds)

            plot_kwds = dict(lw=2, c=color)
            plot_kwds.update(kwds.get("plot_kwds", {}))

            handle = ax.plot(centers, show_y, **plot_kwds)

            if i == 0:
                handles.extend(handle)
                labels.append(model_name)

    for ax, label_name in zip(axes, compare_labels):
        
        ax.set_xlim(0, xlim_upper)

        if ax.is_last_row():
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel(r"S/N")

        else:
            ax.set_xticks([])

        ylim_upper = ylim_uppers.get(label_name,
            250 if label_name.upper() == "TEFF" else 1)
        ax.set_ylim(0, ylim_upper)
        if ax.is_first_col():

            if show_rms:
                ax.set_ylabel(r"${\rm R}.~{\rm M}.~{\rm S}.$")
            else:
                ax.set_ylabel(r"$|\Delta\ell|$")


        ax.yaxis.set_ticks(ax.get_ylim())

        ax.text(0.95, 0.95, latex_label(label_name), transform=ax.transAxes,
            horizontalalignment="right", verticalalignment="top")


    for ax in axes[len(compare_labels):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15, hspace=0.15, top=0.94)
    legend_kwds = dict(loc="upper center", ncol=len(labels), frameon=False,
        fontsize=8)
    legend_kwds.update(kwargs.pop("legend_kwds", {}))
    plt.figlegend(handles, labels, **legend_kwds)

    return fig

