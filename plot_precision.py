

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.cm import Dark2 as colormap
from matplotlib.ticker import MaxNLocator

import thecannon as tc
from astropy.table import Table
from apogee import config
from experiments import aspcap_precision_from_repeat_calibration_visits

#allStar = Table.read("/Users/arc/research/projects/active/the-battery-stars/catalogs/allStar-l31c.2.fits")
allStar = Table.read(os.path.join(config["APOGEE_DR14_DIR"], "allStar-l31c.2.fits"))
allStar["FILE"] = [each.strip() for each in allStar["FILE"]]


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


def plot_rms_given_snr(observed_snr, experiments, compare_labels, square=True, xlim_upper=100,
    ylim_uppers=None, color_offset=0, show_rms=True, line_value=0.1, **kwargs):



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

    line_color = {}
    for i, (ax, label_name) in enumerate(zip(axes, compare_labels)):

        line_color.setdefault(label_name, dict(c="#666666"))

        for j, (model_name, experiment, model_basename, result_basename, kwds) in enumerate(experiments):

            model_path = os.path.join(experiment, model_basename)
            results_path = os.path.join(experiment, result_basename)

            model = tc.CannonModel.read(model_path)
            with open(results_path, "rb") as fp:
                snr, combined_snr, label_differences, filenames = pickle.load(fp)

            # Get results from ASPCAP.
            raise a

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

            color = colormap(kwds.get("color_index", j + color_offset))

            scatter_kwds = dict(s=1, alpha=0.1, c=color)
            scatter_kwds.update(kwds.get("scatter_kwds", {}))

            #ax.scatter(snr, y, **scatter_kwds)

            # Show median and percentiles in each.
            N_bins = kwds.get("N_bins", int(len(snr) / 50))

            bins = equal_histogram_bins(snr, N_bins)
            bins[0] = 0
            centers = bins[:-1] + np.diff(bins)/2.0
            indices = np.digitize(snr, bins)

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

            p_observed_snr, _= np.histogram(observed_snr, bins=bins)
            p_observed_snr = p_observed_snr / np.sum(p_observed_snr)

            obs_rms = np.random.choice(show_y, size=100000, p=p_observed_snr)

            hist_kwds = dict(facecolor=color, alpha=0.5)
            ylim_upper = ylim_uppers.get(label_name,
                250 if label_name.upper() == "TEFF" else 1)

            bins = np.linspace(0, ylim_upper, 20)
            ax.hist(obs_rms, bins=bins, **hist_kwds)

            if np.median(obs_rms) > line_value:
                line_color[label_name].update(c="r", lw=2)


    for ax, label_name in zip(axes, compare_labels):

        ylim_upper = ylim_uppers.get(label_name,
                250 if label_name.upper() == "TEFF" else 1)

        ax.set_xlim(0, ylim_upper)
        if "_" in label_name:
            ax.axvline(line_value, zorder=100, **line_color[label_name])
        ax.xaxis.set_major_locator(MaxNLocator(3))

        if ax.is_last_row():
            ax.set_xlabel(r"$\sigma$")

        ax.yaxis.set_ticks([])

        ax.text(0.95, 0.95, latex_label(label_name), transform=ax.transAxes,
            horizontalalignment="right", verticalalignment="top")


    for ax in axes[len(compare_labels):]:
        ax.set_visible(False)

    fig.tight_layout()


    return fig


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

            color = colormap(kwds.get("color_index", j + color_offset))

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



def plot_precision_relative_to_aspcap(experiments, compare_labels, square=True,
    xlim_upper=None, ylim_uppers=None, color_offset=0, show_rms=True, 
    minimum_combined_snr=0, **kwargs):

    if ylim_uppers is None:
        ylim_uppers = {}

    L = len(compare_labels)
    if square:
        K = int(np.ceil(np.sqrt(L)))
        M = (K - 1) if (K - 1) * K < L else K
        fig, axes = plt.subplots(K, M, figsize=(1.75*K, 1.75*M))

    else:
        fig, axes = plt.subplots(L)

    axes = np.atleast_1d(axes).flatten()


    handles = []
    labels = []

    
    for i, (ax, label_name) in enumerate(zip(axes, compare_labels)):


        for j, (model_name, model_path, results_path, kwds) in enumerate(experiments):

            model = tc.CannonModel.read(model_path)
            with open(results_path, "rb") as fp:
                results = pickle.load(fp)

            visit_snr, combined_snr, visit_snr_labels, \
                aspcap_combined_snr_labels, apogee_ids = results

            try:
                label_index = model.vectorizer.label_names.index(label_name)

            except ValueError:
                print("Could not find {} in model stored at {}".format(
                    label_name, model_path))
                continue


            # We need some serious quality control here.
            qc = np.all(aspcap_combined_snr_labels > -9999, axis=1) \
               * (combined_snr >= minimum_combined_snr) \
               * np.all(visit_snr_labels > -9999, axis=1)

            # There are some stars that are obviously incorrect in ASPCAP
            # calibration things, but here I don't have the flags to be able
            # to easily discriminate.

            # TODO: Revisit this using ASPCAP flags.
            qc *= (np.abs(aspcap_combined_snr_labels[:, 0] - visit_snr_labels[:, 0]) < 500)


            """

            # Deal with ASPCAP values being -10000
            unusable = np.abs(aspcap_combined_snr_labels[:, label_index]) > 9000
            # IT'S OVER 9000 WHAT THE ACTUAL FUCK.

            if any(unusable):
                print("Discarding {} measurements because ASPCAP provides no values".format(sum(unusable)))
                visit_snr = visit_snr[~unusable]
                combined_snr = combined_snr[~unusable]
                visit_snr_labels = visit_snr_labels[~unusable]
                aspcap_combined_snr_labels = aspcap_combined_snr_labels[~unusable]
                apogee_ids = apogee_ids[~unusable]

            use = combined_snr >= minimum_combined_snr
            print("Discarding {} because did not reach the minimum combined SNR of {}".format(
                sum(~use), minimum_combined_snr))
            visit_snr = visit_snr[use]
            combined_snr = combined_snr[use]
            visit_snr_labels = visit_snr_labels[use]
            aspcap_combined_snr_labels = aspcap_combined_snr_labels[use]
            apogee_ids = apogee_ids[use]
            """

            visit_snr = visit_snr[qc]
            combined_snr = combined_snr[qc]
            visit_snr_labels = visit_snr_labels[qc]
            aspcap_combined_snr_labels = aspcap_combined_snr_labels[qc]
            apogee_ids = apogee_ids[qc]

            y = np.abs(aspcap_combined_snr_labels[:, label_index] - visit_snr_labels[:, label_index])

            color = colormap(kwds.get("color_index", j + color_offset))

            scatter_kwds = dict(s=1, alpha=0.1, c=color)
            scatter_kwds.update(kwds.get("scatter_kwds", {}))

            ax.scatter(visit_snr, y, **scatter_kwds)

            if np.any(y > 5000):
                raise wtf

            # Show median and percentiles in each.
            N_bins = 50 # kwds.get("N_bins", int(len(visit_snr) / 50))
            percentiles = kwds.get("percentiles", [16, 50, 84])

            bins = equal_histogram_bins(visit_snr, N_bins)
            bins[0] = 0
            centers = bins[:-1] + np.diff(bins)/2.0
            indices = np.digitize(visit_snr, bins) - 1

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

        if xlim_upper is not None:
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
