#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of demcompare
# (see https://github.com/CNES/demcompare).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Stats module of dsm_compare offers routines
for stats computation and plot viewing
"""

# Standard imports
import collections
import copy
import csv
import json
import logging
import math
import os
import traceback
from typing import Dict, List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as mpl_pyplot

# Third party imports
import numpy as np
import xarray as xr
from astropy import units as u
from matplotlib import gridspec
from scipy.optimize import curve_fit

# DEMcompare imports
from .img_tools import read_image, read_img_from_array, save_tif
from .output_tree_design import get_out_file_path
from .partition import FusionPartition, NotEnoughDataToPartitionError, Partition


class NoPointsToPlot(Exception):
    """Specific NoPointsToPlot Exception Class creation"""


def compute_stats_array(
    cfg: Dict,
    dem: np.ndarray,
    ref: np.ndarray,
    dem_nodata: Union[int, float, None] = None,
    ref_nodata: Union[int, float, None] = None,
    display: bool = False,
    final_json_file: Union[str, None] = None,
):
    """
    Compute Stats from numpy arrays

    :param cfg: configuration dictionary
    :type cfg: dict
    :param dem: dem raster
    :type dem: numpy array
    :param ref: reference dem raster to be coregistered to dem
    :type ref: numpy array
    :param dem_nodata: inodata value in dem
    :type dem_nodata: int, float or None
    :param ref_nodata: nodata value in ref
    :type ref_nodata: int, float or None
    :param display: choose between plot show and plot save
    :type display: boolean
    :param final_json_file: filename of final_cfg
    :type final_json_file: str or None
    :return: None
    """

    if "stats_opts" not in cfg:
        cfg["stats_opts"] = {}

    if "to_be_classification_layers" not in cfg["stats_opts"]:
        cfg["stats_opts"]["to_be_classification_layers"] = {}

    if "classification_layers" not in cfg["stats_opts"]:
        cfg["stats_opts"]["classification_layers"] = {}

    if "stats_results" not in cfg:
        cfg["stats_results"] = {}

    # default config
    cfg["plani_results"] = {}
    cfg["plani_results"]["dx"] = 1
    cfg["plani_results"]["dy"] = 1

    cfg["alti_results"] = {}
    cfg["alti_results"]["rectifiedRef"] = {}
    cfg["alti_results"]["rectifiedRef"]["nb_valid_points"] = 10
    cfg["alti_results"]["rectifiedRef"]["nb_points"] = 10
    cfg["alti_results"]["rectifiedDEM"] = {}
    cfg["alti_results"]["rectifiedDEM"]["nb_valid_points"] = 10
    cfg["alti_results"]["rectifiedDEM"]["nb_points"] = 10

    cfg["stats_opts"]["alti_error_threshold"] = {}
    cfg["stats_opts"]["alti_error_threshold"]["value"] = 0
    cfg["stats_opts"]["plot_real_hists"] = False
    cfg["stats_opts"]["remove_outliers"] = False

    dem_a3d = read_img_from_array(dem, no_data=dem_nodata)
    ref_a3d = read_img_from_array(ref, no_data=ref_nodata)

    final_dh = dem_a3d["im"].data - ref_a3d["im"].data
    final_dh_a3d = read_img_from_array(final_dh)

    if final_json_file is None:
        final_json_file = cfg["outputDir"] + "/final_stats.json"

    alti_diff_stats(
        cfg,
        dem_a3d,
        ref_a3d,
        final_dh_a3d,
        display=display,
        remove_outliers=cfg["stats_opts"]["remove_outliers"],
        geo_ref=False,
    )
    # save results
    with open(final_json_file, "w", encoding="utf8") as outfile:
        json.dump(cfg, outfile, indent=2)


def gaus(x, a, x_zero, sigma):
    """
    Gauss math function

    :param x:
    :type x:
    :param a:
    :type a:
    :param x_zero:
    :type x_zero:
    :param sigma:
    :type sigma:
    :return: gauss function
    :rtype:
    """
    return a * np.exp(-((x - x_zero) ** 2) / (2 * sigma**2))


def round_up(x, y):
    """Round up math function

    :param x:
    :type x:
    :param y:
    :type y:
    :return:
    :rtype:
    """
    return int(math.ceil((x / float(y)))) * y


def get_nonan_mask(
    array: np.ndarray, no_data_value: Union[int, None] = None
) -> np.ndarray:
    """
    Get no data and nan mask value

    :param array: input array to get the mask from
    :type array: np.ndarray
    :param no_data_value: no data value considered. Default: None
    :type no_data_value: int or None
    :return: nan and no_data_value if exists mask on array.
    :rtype: np.ndarray
    """
    if no_data_value is None:
        return np.apply_along_axis(lambda x: (~np.isnan(x)), 0, array)
    # else no_data_value exists
    return np.apply_along_axis(
        lambda x: (~np.isnan(x)) * (x != no_data_value), 0, array
    )


def get_outliers_free_mask(
    array: np.ndarray, no_data_value: Union[int, None] = None
) -> np.ndarray:
    """
    Get outliers free mask (array of True where value is no outlier) with
    values outside (mu + 3 sigma) and (mu - 3 sigma).
    Nan and no_data_value are not considered in mu and sigma computation.

    :param array: input array to get the mask from
    :type array: np.ndarray
    :param no_data_value: no data value considered. Default: None
    :type no_data_value: int or None
    :return: outliers free mask (array of True where value is no outlier)
    :rtype: np.ndarray
    """
    # pylint: disable=singleton-comparison
    no_data_free_mask = get_nonan_mask(array, no_data_value)
    array_without_nan = array[np.where(no_data_free_mask == True)]  # noqa: E712
    mu = np.mean(array_without_nan)
    sigma = np.std(array_without_nan)
    return np.apply_along_axis(
        lambda x: (x > mu - 3 * sigma) * (x < mu + 3 * sigma), 0, array
    )


def create_mode_masks(alti_map: xr.Dataset, partitions_sets_masks=None):
    """
    Compute Masks for every required modes :

    the 'standard' mode: where the mask
    stands for nan values inside the error image
    with the nan values inside the ref_support_desc
    when do_classification is on & it also stands for outliers free values

    the 'coherent-classification' mode:
    which is the 'standard' mode where only the pixels
    for which both sets (dsm and reference) are coherent

    the 'incoherent-classification' mode:
    which is 'coherent-classification' complementary

    Note that 'coherent-classification'
    and 'incoherent-classification' mode masks
    can only be computed if len(partitions_sets_masks)==2

    :param alti_map: alti differences
    :type alti_map: xr.Dataset
    :param partitions_sets_masks: [] (master and/or slave dsm)
            of [] of boolean array (sets for each dsm)
    :type partitions_sets_masks: List[bool]
    :return: list of masks, associated modes, and error_img read as array
    :rtype: List[np.ndarray]
    """

    mode_names = []
    mode_masks = []

    # Starting with the 'standard' mask
    mode_names.append("standard")
    # -> remove alti_map nodata indices
    mode_masks.append(
        get_nonan_mask(alti_map["im"].data, alti_map.attrs["no_data"])
    )
    # -> remove nodata indices for every partitioning image
    if partitions_sets_masks:
        for partition_img in partitions_sets_masks:
            # for a given partition,
            # nan values are flagged False for all sets hence
            # np.any return True for a pixel
            # if it belongs to at least one set (=it this is not a nodata pixel)
            partition_nonan_mask = np.any(partition_img, axis=0)
            mode_masks[0] *= partition_nonan_mask

    # Carrying on with potentially
    # the cross classification (coherent & incoherent) masks
    if len(partitions_sets_masks) == 2:
        # case where there's a classification img to partition
        # from for both master & slave dsm
        mode_names.append("coherent-classification")
        # Combine pairs of sets together
        # (meaning first partition first set with second partition first set)
        # -> then for each single class / set,
        #    we know which pixels are coherent between both partitions
        # -> combine_sets[0].shape[0] = number of sets (classes)
        # -> combine_sets[0].shape[1] = number of pixels inside a single DSM
        partition_imgs = partitions_sets_masks
        combine_sets = np.array(
            [
                partition_imgs[0][set_idx][:] == partition_imgs[1][set_idx][:]
                for set_idx in range(0, len(partition_imgs[0]))
            ]
        )
        coherent_mask = np.all(combine_sets, axis=0)
        mode_masks.append(mode_masks[0] * coherent_mask)

        # Then the incoherent one
        mode_names.append("incoherent-classification")
        mode_masks.append(mode_masks[0] * ~coherent_mask)

    return mode_masks, mode_names


def create_masks(
    alti_map: xr.Dataset,
    do_classification: bool = False,
    ref_support: xr.Dataset = None,
    do_cross_classification: bool = False,
    ref_support_classified_desc: Dict = None,
    remove_outliers=True,
) -> List[np.ndarray]:
    """
    Compute Masks for every required modes :

    the 'standard' mode
    where the mask stands for nan values
    inside the error image with the nan values
    inside the ref_support_desc when do_classification is on
    & it also stands for outliers free values

    the 'coherent-classification' mode
    which is the 'standard' mode where only the
    pixels for which both sets
    (dsm and reference) are coherent

    the 'incoherent-classification' mode
    which is 'coherent-classification' complementary

    :param alti_map: alti differences
    :type alti_map: xr.Dataset
    :param do_classification: wether or not the classification is activated
    :type do_classification: bool
    :param ref_support: reference support dataset
    :type ref_support: xr.Dataset
    :param do_cross_classification: whether or not
            the cross classification is activated
    :type do_cross_classification: bool
    :param ref_support_classified_desc: dict with
            'path' and 'nodata' keys for the ref support image classified
    :type ref_support_classified_desc: dict
    :param remove_outliers: boolean, set to
            True (default) to return a no_outliers mask
    :type remove_outliers: bool
    :return: list of masks, associated modes, and error_img read as array
    :rtype: List[np.ndarray]
    """

    modes = []
    masks = []

    # Starting with the 'standard' mask with no nan values
    modes.append("standard")
    masks.append(get_nonan_mask(alti_map["im"].data, alti_map.attrs["no_data"]))

    # Create no outliers mask if required
    no_outliers = None
    if remove_outliers:
        no_outliers = get_outliers_free_mask(
            alti_map["im"].data, alti_map.attrs["no_data"]
        )

    # If the classification is on then we also consider ref_support nan values
    if do_classification:
        masks[0] *= get_nonan_mask(
            ref_support["im"].data, ref_support.attrs["no_data"]
        )

    # Carrying on with potentially the cross classification masks
    if do_classification and do_cross_classification:
        modes.append("coherent-classification")

        ref_support_classified_val = read_image(
            ref_support_classified_desc["path"], band=4
        )
        # so we get rid of what are actually 'nodata'
        # and incoherent values as well
        coherent_mask = get_nonan_mask(
            ref_support_classified_val, ref_support_classified_desc["nodata"][0]
        )
        masks.append(masks[0] * coherent_mask)

        # Then the incoherent one
        modes.append("incoherent-classification")
        masks.append(masks[0] * ~coherent_mask)

    return masks, modes, no_outliers


def stats_computation(
    array: np.ndarray, list_threshold: List[int] = None
) -> Dict:
    """
    Compute stats for a specific array

    :param array: numpy array
    :type array: np.ndarray
    :param list_threshold: list, defines thresholds
            to be used for pixels above thresholds ratio computation
    :type list_threshold: List[int]
    :return: dict with stats name and values
    :rtype: Dict
    """
    if array.size:
        res = {
            "nbpts": array.size,
            "max": round(float(np.max(array)), 5),
            "min": round(float(np.min(array)), 5),
            "mean": round(float(np.mean(array)), 5),
            "std": round(float(np.std(array)), 5),
            "rmse": round(float(np.sqrt(np.mean(array * array))), 5),
            "median": round(float(np.nanmedian(array)), 5),
            "nmad": round(
                float(
                    1.4826 * np.nanmedian(np.abs(array - np.nanmedian(array)))
                ),
                5,
            ),
            "sum_err": round(float(np.sum(array)), 5),
            "sum_err.err": round(float(np.sum(array * array)), 5),
        }
        if list_threshold:
            res["ratio_above_threshold"] = {
                threshold: round(
                    float(np.count_nonzero(array > threshold))
                    / float(array.size),
                    5,
                )
                for threshold in list_threshold
            }
        else:
            res["ratio_above_threshold"] = {"none": np.nan}
    else:
        res = {
            "nbpts": array.size,
            "max": np.nan,
            "min": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "rmse": np.nan,
            "median": np.nan,
            "nmad": np.nan,
            "sum_err": np.nan,
            "sum_err.err": np.nan,
        }
        if list_threshold:
            res["ratio_above_threshold"] = {
                threshold: np.nan for threshold in list_threshold
            }
        else:
            res["ratio_above_threshold"] = {"none": np.nan}
    return res


def get_stats(
    dz_values: np.ndarray,
    to_keep_mask: np.ndarray = None,
    sets: List[np.ndarray] = None,
    sets_labels: List[str] = None,
    sets_names: List[str] = None,
    list_threshold: List[int] = None,
    outliers_free_mask=None,
) -> List[Dict]:
    """
    Get Stats for a specific array, considering
    potentially subsets of it

    :param dz_values: errors
    :type dz_values: np.ndarray
    :param to_keep_mask: boolean mask with
            True values for pixels to use
    :type to_keep_mask: List[bool]
    :param sets: list of sets (boolean arrays
            that indicate which class a pixel belongs to)
    :type sets: List[np.ndarray]
    :type sets_labels: label associated to the sets
    :param sets_labels: List[str]
    :param sets_names: name associated to the sets
    :type sets_names: List[str]
    :param list_threshold: list, defines thresholds to
            be used for pixels above thresholds ratio computation
    :type list_threshold: List[int]
    :param outliers_free_mask:
    :type outliers_free_mask:
    :return: list of dictionary (set_name, nbpts,
             %(out_of_all_pts), max, min, mean, std, rmse, ...)
    :rtype: List[Dict]
    """
    # pylint: disable=singleton-comparison

    def nighty_percentile(array):
        """
        Compute the maximal error for the 90% smaller errors

        :param array:
        :return:
        """
        if array.size:
            return np.nanpercentile(np.abs(array - np.nanmean(array)), 90)
        # else:
        return np.nan

    # Init
    output_list = []
    nb_total_points = dz_values.size
    # - if a mask is not set,
    # we set it with True values only so that it has no effect
    if to_keep_mask is None:
        to_keep_mask = np.ones(dz_values.shape)
    if outliers_free_mask is None:
        outliers_free_mask = np.ones(dz_values.shape)

    # Computing first set of values with all pixels considered
    # -except the ones masked or the outliers-
    output_list.append(
        stats_computation(
            dz_values[
                np.where(
                    to_keep_mask * outliers_free_mask == True  # noqa: E712
                )
            ],
            list_threshold,
        )
    )
    # - we add standard information for later use
    output_list[0]["set_label"] = "all"
    output_list[0]["set_name"] = "All classes considered"
    output_list[0]["%"] = round(
        ((100 * float(output_list[0]["nbpts"]) / float(nb_total_points))), 5
    )
    # - we add computation of nighty percentile
    # (of course we keep outliers for that so we use dz_values as input array)
    output_list[0]["90p"] = round(
        nighty_percentile(
            dz_values[np.where(to_keep_mask == True)]  # noqa: E712
        ),
        5,
    )

    # Computing stats for all sets (sets are a partition of all values)
    if sets is not None and sets_labels is not None and sets_names is not None:
        for set_idx, set_item in enumerate(sets):
            set_item = set_item * to_keep_mask * outliers_free_mask

            data = dz_values[np.where(set_item == True)]  # noqa: E712
            output_list.append(stats_computation(data, list_threshold))
            output_list[set_idx + 1]["set_label"] = sets_labels[set_idx]
            output_list[set_idx + 1]["set_name"] = sets_names[set_idx]
            output_list[set_idx + 1]["%"] = round(
                (
                    100
                    * float(output_list[set_idx + 1]["nbpts"])
                    / float(nb_total_points)
                ),
                5,
            )
            output_list[set_idx + 1]["90p"] = round(
                nighty_percentile(
                    dz_values[
                        np.where(
                            (sets[set_idx] * to_keep_mask) == True  # noqa: E712
                        )
                    ]
                ),
                5,
            )

    return output_list


def dem_diff_plot(
    dem_diff: xr.Dataset,
    title: str = "",
    plot_file: str = "dem_diff.png",
    display: bool = False,
):
    """
    Simple img show after outliers removal

    :param dem_diff: difference dem
    :type dem_diff: xr.Dataset
    :param title: plot title
    :type title: str
    :param plot_file: path and name for
            the saved plot (used when display if False)
    :type plot_file: str
    :param display: boolean, set to True if display is on,
            otherwise the plot is saved to plot_file location
    :type display: bool
    """
    # Init mu and sigma from data to focus on little values
    mu = np.nanmean(dem_diff["im"].data)
    sigma = np.nanstd(dem_diff["im"].data)

    # Plot
    fig, fig_ax = mpl_pyplot.subplots(figsize=(7.0, 8.0))
    fig_ax.set_title(title, fontsize="large")
    im1 = fig_ax.imshow(
        dem_diff["im"].data, cmap="terrain", vmin=mu - sigma, vmax=mu + sigma
    )
    fig.colorbar(im1, label="Elevation differences (m)")
    fig.text(
        0.15,
        0.15,
        "Image diff view: [Min, Max]=[{:.2f}, {:.2f}]".format(
            mu - sigma, mu + sigma
        ),
        fontsize="medium",
    )

    #
    # Show or Save
    #
    if display is False:
        mpl_pyplot.savefig(plot_file, dpi=100, bbox_inches="tight")
    else:
        mpl_pyplot.show()
    mpl_pyplot.close()


def dem_diff_cdf_plot(
    dem_diff: xr.Dataset,
    title: str = "",
    plot_file: str = "dem_diff_cdf.png",
    bin_step: float = 0.1,
    display: bool = False,
):
    """
    Simple img values absolute cdf view truncated
    by max_diff

    :param dem_diff: difference dem,
    :type dem_diff: xarray Dataset
    :param title: plot title
    :type title: str
    :param plot_file: path and name for the
            saved plot (used when display if False)
    :type plot_file: str
    :param bin_step: bin size
    :type bin_step: float
    :param display: set to True if display is on,
            otherwise the plot is saved to plot_file location
    :type display: bool
    """
    # Get plot_file file base
    plot_file_base = os.path.splitext(plot_file)[0]

    # Generate absolute values array
    abs_dem_diff = np.abs(dem_diff["im"].data)

    # Get max diff from data
    max_diff = np.nanmax(abs_dem_diff)

    # Count nb nan
    nb_nans = np.sum(np.isnan(abs_dem_diff))
    nb_pixels = abs_dem_diff.shape[0] * abs_dem_diff.shape[1]

    # Get bins number for histogram
    nb_bins = int(max_diff / bin_step)

    # getting data of the histogram
    hist, bins_count = np.histogram(
        abs_dem_diff, bins=nb_bins, range=(0, max_diff), density=True
    )

    # Normalized Probability Density Function of the histogram
    pdf = hist / sum(hist)

    # Generate Cumulative Probability Function
    cdf = np.cumsum(pdf)

    # Save cdf in csv in same base file name.
    with open(
        plot_file_base + ".csv", "w", newline="", encoding="utf8"
    ) as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["Bins", "CDF values"])
        writer.writerows(zip(bins_count, cdf))

    # Plot
    fig, fig_ax = mpl_pyplot.subplots()
    fig_ax.set_title(title, fontsize="large")
    fig_ax.plot(bins_count[1:], cdf, label="CDF")

    # tidy up the figure and add axes titles
    fig_ax.set_xlabel(
        "Full absolute elevation differences (m) "
        "\nmax_diff={} nb_bins={}"
        "\nnb_pixels={} nb_nans={}".format(
            max_diff, nb_bins, nb_pixels, nb_nans
        ),
        fontsize="medium",
    )
    fig_ax.set_ylabel("Cumulative Probability [0,1]", fontsize="medium")
    fig_ax.set_ylim(0, 1.05)
    fig_ax.grid(True)

    # Show or Save
    if display is False:
        fig.savefig(plot_file, dpi=100, bbox_inches="tight")
    else:
        fig.show()
    mpl_pyplot.close()


def dem_diff_pdf_plot(
    dem_diff: xr.Dataset,
    title: str,
    plot_file: str,
    display: bool,
    bin_step: float = 0.2,
    width: float = 0.7,
):
    """
    Computes the difference histogram.

    The .tif output histogram plot is centered around 0
    and bounded over [- |percentile98|, |percentile98|]
    The output csv file has all the data.

    :param dem_diff: difference dem
    :type dem_diff: xarray Dataset
    :param title: plot title
    :type title: str
    :param plot_file: path and name for the saved
            plot (used when display if False)
    :type plot_file: str
    :param display: bset to True if display is on,
            otherwise the plot is saved to plot_file location
    :type display: bool
    :param bin_step: bin size
    :type bin_step: float
    :param width: plot's bin width
    :type bin_step: float
    :return: None
    """

    # Get plot_file file base
    plot_file_base = os.path.splitext(plot_file)[0]

    # Generate values array
    dem_diff = dem_diff["im"].data

    # Histogram plot creation.
    # The histogram is centered around 0
    # and bounded over [- |percentile98|, |percentile98|]
    p98 = np.abs(np.nanpercentile(dem_diff, 98))

    hist, bins = np.histogram(
        dem_diff[~np.isnan(dem_diff)],
        bins=np.arange(-p98, p98, bin_step),
    )

    # Normalized Probability Density Function of the histogram
    pdf = hist / sum(hist)

    # Define histogram's bins width
    width = width * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    fig0 = mpl_pyplot.figure()
    # Define axes labels and title
    fig0_ax = fig0.add_subplot(111)
    fig0_ax.set_title(title, fontsize=12)
    fig0_ax.set_xlabel(
        "Elevation difference (m) from - |p98| to |p98|", fontsize=12
    )
    fig0_ax.set_ylabel("Normalized frequency", fontsize=12)
    mpl_pyplot.grid(True)
    mpl_pyplot.bar(center, pdf, align="center", width=width)

    # Generate histogram with all data to save in csv
    max_diff = np.nanmax(dem_diff)
    full_hist, bins = np.histogram(
        dem_diff[~np.isnan(dem_diff)],
        bins=np.arange(-max_diff, max_diff, bin_step),
    )

    # Normalized Probability Density Function of the histogram
    pdf = full_hist / sum(full_hist)

    # Save pdf in csv in same base file name.
    with open(
        plot_file_base + ".csv", "w", newline="", encoding="utf8"
    ) as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["Bins", "Normalized PDF values"])
        writer.writerows(zip(bins, pdf))

    if display is False:
        # Save histogram to .png image
        mpl_pyplot.savefig(
            plot_file,
            dpi=100,
            bbox_inches="tight",
        )
    else:
        mpl_pyplot.show()
    mpl_pyplot.close(fig0)


def plot_histograms(  # noqa: C901
    input_array: np.ndarray,
    bin_step: float = 0.1,
    to_keep_mask: np.ndarray = None,
    sets: List = None,
    sets_labels: np.ndarray = None,
    sets_colors: np.ndarray = None,
    plot_title: str = "",
    outplotdir: str = ".",
    outhistdir: str = ".",
    save_prefix: str = "",
    display: bool = False,
    plot_real_hist: bool = False,
):
    """
    Creates a histogram plot for all sets given and saves them on disk.

    Note :
    If more than one set is given, than all the remaining sets are supposed
    to partitioned the first one.
    Hence, in the contribution plot the first set is not considered and all
    percentage are computed in regards of the number of points
    within the first set (which is supposed to contain them all)

    :param input_array: data to plot
    :type input_array: np.ndarray
    :param bin_step: histogram bin step
    :type bin_step: float
    :param to_keep_mask: boolean mask with True values for pixels to use
    :type to_keep_mask: np.ndarray
    :param sets: list of sets (boolean arrays that
            indicate which class a pixel belongs to)
    :type sets: List
    :param set_labels: name associated to the sets
    :type set_labels: np.ndarray
    :param sets_colors: color set for plotting
    :type sets_colors: np.ndarray
    :param plot_title: plot title
    :type plot_title: str
    :param outplotdir: directory where histograms are to be saved
    :type outplotdir: str
    :param outhistdir: directory where
            histograms (as numpy files) are to be saved
    :type outhistdir: str
    :param save_prefix: prefix to the histogram files saved by this method
    :type save_prefix: str
    :param display: set to False to save plot instead of actually plotting them
    :type display: bool
    :param plot_real_hist: plot or save (see display param) real histograms
    :type plot_real_hist: bool
    :return: list saved files
    """
    # pylint: disable=singleton-comparison

    saved_files = []
    saved_labels = []
    saved_colors = []
    #
    # Plot initialization
    #

    mpl.rc("font", size=6)
    if display:
        mpl.use("TkAgg")

    # -> bins should rely on [-A;A],A being the higher absolute error value
    # (all histograms rely on the same bins range)
    if to_keep_mask is not None:
        if input_array[np.where(to_keep_mask == True)].size != 0:  # noqa: E712
            borne = np.max(
                [
                    abs(
                        np.nanmin(
                            input_array[
                                np.where(to_keep_mask == True)  # noqa: E712
                            ]
                        )
                    ),
                    abs(
                        np.nanmax(
                            input_array[
                                np.where(to_keep_mask == True)  # noqa: E712
                            ]
                        )
                    ),
                ]
            )
        else:
            raise NoPointsToPlot
    else:
        borne = np.max(
            [abs(np.nanmin(input_array)), abs(np.nanmax(input_array))]
        )
    bins = np.arange(
        -round_up(borne, bin_step),
        round_up(borne, bin_step) + bin_step,
        bin_step,
    )

    np.savetxt(
        os.path.join(outhistdir, save_prefix + "bins" + ".txt"),
        [bins[0], bins[len(bins) - 1], bin_step],
    )

    # Figure 1 : One plot of Normalized Histograms
    # -> set figures shape, titles and axes
    if plot_real_hist:
        fig1 = mpl_pyplot.figure(1, figsize=(7.0, 8.0))
        fig1.suptitle(plot_title)
        fig1_ax = fig1.add_subplot(111)
        fig1_ax.set_title("Data shown as normalized histograms")
        fig1_ax.set_xlabel("Errors (meter)")
        data = []
        full_color = []
        for set_idx, _ in enumerate(sets):
            # -> restricts to input data
            if to_keep_mask is not None:
                sets[set_idx] = sets[set_idx] * to_keep_mask
            # print(
            #     "}}}}}}}}}}}}}}}}}}}}}}}}}}}}}} plot_histograms() : ",
            #     np.where(sets[set_idx] == True),  # noqa: E712
            # )
            data.append(
                input_array[np.where(sets[set_idx] == True)]  # noqa: E712
            )
            full_color.append(sets_colors[set_idx])
        fig1_ax.hist(
            data,
            density=True,
            label=sets_labels,
            histtype="step",
            color=full_color,
        )
        fig1_ax.legend()
        if display is False:
            fig1.savefig(
                os.path.join(
                    outplotdir,
                    "AltiErrors_RealHistrograms_" + save_prefix + ".png",
                ),
                dpi=100,
                bbox_inches="tight",
            )
        else:
            mpl_pyplot.figure(1)
            mpl_pyplot.show()

        # Check fig1 to close
        mpl_pyplot.figure(1)
        mpl_pyplot.close()
        # TODO : add in saved_files return ?

    # Figure 2 : Two plots fitted by gaussian histograms & classes contributions
    #    -> set figure shape, titles and axes
    fig2 = mpl_pyplot.figure(2, figsize=(7.0, 8.0))
    fig2.suptitle(plot_title)
    grid = gridspec.GridSpec(
        1, 2, width_ratios=[10, 1]
    )  # Specifies the geometry of the grid that a subplot will be placed
    # Create gaussian histograms errors axe
    fig2_ax_errors = fig2.add_subplot(grid[0])
    fig2_ax_errors.set_title("Errors fitted by a gaussian")
    fig2_ax_errors.set_xlabel("Errors (in meter)")
    # Create classes contributions axe
    fig2_ax_classes = fig2.add_subplot(grid[1])
    fig2_ax_classes.set_title("Classes contributions")
    fig2_ax_classes.set_xticks(np.arange(1), minor=False)

    # Generate plots axes with data
    cumulative_percent = 0
    set_zero_size = 0
    if sets is not None and sets_labels is not None and sets_colors is not None:
        for set_idx, _ in enumerate(sets):
            # -> restricts to input data
            if to_keep_mask is not None:
                sets[set_idx] = sets[set_idx] * to_keep_mask
            data = input_array[np.where(sets[set_idx] == True)]  # noqa: E712

            # -> empty data is not plotted
            if data.size:
                mean = np.mean(data)
                std = np.std(data)
                nb_points_as_percent = (
                    100 * float(data.size) / float(input_array.size)
                )
                if set_idx != 0:
                    set_contribution = (
                        100 * float(data.size) / float(set_zero_size)
                    )
                    cumulative_percent += set_contribution
                else:
                    set_zero_size = data.size

                try:
                    n, bins = np.histogram(data, bins=bins, density=True)
                    fit_result = curve_fit(
                        gaus,
                        bins[0 : bins.shape[0] - 1] + int(bin_step / 2),
                        n,
                        p0=[1, mean, std],
                    )
                    # get popt and avoid unbalanced-tuple-unpacking message
                    popt, _ = fit_result[:2]
                    fig2_ax_errors.plot(
                        np.arange(
                            bins[0], bins[bins.shape[0] - 1], bin_step / 10
                        ),
                        gaus(
                            np.arange(
                                bins[0], bins[bins.shape[0] - 1], bin_step / 10
                            ),
                            *popt
                        ),
                        color=sets_colors[set_idx],
                        linewidth=1,
                        label=" ".join(
                            [
                                sets_labels[set_idx],
                                r"$\mu$ {0:.2f}m".format(mean),
                                r"$\sigma$ {0:.2f}m".format(std),
                                "{0:.2f}% points".format(nb_points_as_percent),
                            ]
                        ),
                    )
                    if set_idx != 0:
                        # 1 is the x location and 0.05 is the width
                        # (label is not printed)
                        fig2_ax_classes.bar(
                            1,
                            set_contribution,
                            0.05,
                            color=sets_colors[set_idx],
                            bottom=cumulative_percent - set_contribution,
                            label="test",
                        )

                        fig2_ax_classes.text(
                            1,
                            cumulative_percent - 0.5 * set_contribution,
                            "{0:.2f}".format(set_contribution),
                            weight="bold",
                            horizontalalignment="left",
                        )
                except RuntimeError:
                    print(
                        "No fitted gaussian plot "
                        "created as curve_fit failed to converge"
                    )

                # save outputs (plot files and name of labels kept)
                saved_labels.append(sets_labels[set_idx])
                saved_colors.append(sets_colors[set_idx])
                saved_file = os.path.join(
                    outhistdir, save_prefix + str(set_idx) + ".npy"
                )
                saved_files.append(saved_file)
                np.save(saved_file, n)

    # Set aggregated legend
    fig2_ax_errors.legend(loc="upper left")

    # Figure 2 Plot save or show
    if display is False:
        fig2.savefig(
            os.path.join(
                outplotdir,
                "AltiErrors-Histograms_FittedWithGaussians_"
                + save_prefix
                + ".png",
            ),
            dpi=100,
            bbox_inches="tight",
        )
    else:
        mpl_pyplot.figure(2)
        mpl_pyplot.show()

    # Check figure2 to close
    mpl_pyplot.figure(2)
    mpl_pyplot.close()

    return saved_files, saved_labels, saved_colors


def save_results(
    output_json_file: str,
    stats_list: List[str],
    labels_plotted: List[str] = None,
    plot_files: List[str] = None,
    plot_colors: List[str] = None,
    to_csv: bool = False,
):
    """
    Saves stats into specific json file (and optionally to csv file)

    :param output_json_file: file in which to save
    :type output_json_file: str
    :param stats_list: all the stats to save (one element per label)
    :type stats_list: List[str]
    :param labels_plotted: list of labels plotted
    :type labels_plotted: List[str]
    :param plot_files: list of plot files associdated to the labels_plotted
    :type plot_files: List[str]
    :param plot_colors: list of plot colors associated to the labels_plotted
    :type plot_colors: List[str]
    :param to_csv: boolean, set to True
            to save to csv format as well (default False)
    :type to_csv: List[str]
    :return:
    """

    results = {}
    for stats_index, stats_elem in enumerate(stats_list):
        results[str(stats_index)] = stats_elem
        if (
            labels_plotted is not None
            and plot_files is not None
            and plot_colors is not None
        ):
            if stats_elem["set_label"] in labels_plotted:
                try:
                    results[str(stats_index)]["plot_file"] = plot_files[
                        labels_plotted.index(stats_elem["set_label"])
                    ]
                    results[str(stats_index)]["plot_color"] = tuple(
                        np.round(
                            plot_colors[
                                labels_plotted.index(stats_elem["set_label"])
                            ],
                            decimals=5,
                        )
                    )
                except Exception:
                    print(
                        "Error: plot_files and plot_colors "
                        "should have same dimension as labels_plotted"
                    )
                    raise

    with open(output_json_file, "w", encoding="utf8") as outfile:
        json.dump(results, outfile, indent=4)

    if to_csv:
        # Print the merged results into a csv file
        # with only "important" fields and extended fieldnames
        # - create filename
        csv_filename = os.path.join(
            os.path.splitext(output_json_file)[0] + ".csv"
        )
        # - fill csv_results with solely the filed required
        csv_results = collections.OrderedDict()
        for set_idx in range(0, len(results)):
            key = str(set_idx)
            csv_results[key] = collections.OrderedDict()
            csv_results[key]["Set Name"] = results[key]["set_name"]
            csv_results[key]["% Of Valid Points"] = results[key]["%"]
            csv_results[key]["Max Error"] = results[key]["max"]
            csv_results[key]["Min Error"] = results[key]["min"]
            csv_results[key]["Mean Error"] = results[key]["mean"]
            csv_results[key]["Error std"] = results[key]["std"]
            csv_results[key]["RMSE"] = results[key]["rmse"]
            csv_results[key]["Median Error"] = results[key]["median"]
            csv_results[key]["NMAD"] = results[key]["nmad"]
            csv_results[key]["90 percentile"] = results[key]["90p"]
        # - writes the results down as csv format
        with open(csv_filename, "w", encoding="utf8") as csvfile:
            fieldnames = list(csv_results["0"].keys())
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC
            )

            writer.writeheader()
            for set_item in csv_results:
                writer.writerow(csv_results[set_item])


def create_partitions(
    dsm: xr.Dataset,
    ref: xr.Dataset,
    cfg: Dict,
    geo_ref: bool = True,
):
    """
    Create or adapt all classification supports for the stats.
    If the support is a slope,it's transformed into a classification support.
    :param dsm: xarray Dataset, dsm
    :param ref: xarray Dataset, coregistered ref
    :param cfg: cfg
    :param geo_ref: boolean, set to False if images are not georeferenced
    :type geo_ref: bool
    :return: dict, with partitions information {'
    """
    output_dir = cfg["outputDir"]
    stats_opts = cfg["stats_opts"]
    to_be_clayers = stats_opts["to_be_classification_layers"].copy()
    clayers = stats_opts["classification_layers"].copy()

    logging.debug(
        "list of to be classification layers: {}".format(to_be_clayers)
    )
    logging.debug("list of already classification layers: {}".format(clayers))

    # Create obj partition
    partitions = []
    for layer_name, tbclayer in to_be_clayers.items():
        try:
            partitions.append(
                Partition(
                    layer_name,
                    "to_be_classification_layers",
                    dsm,
                    ref,
                    output_dir,
                    **tbclayer
                )
            )
        except Exception as error:
            traceback.print_exc()
            print(
                (
                    "Cannot create partition for {}:{} -> {}".format(
                        layer_name, tbclayer, error
                    )
                )
            )

    for layer_name, clayer in clayers.items():
        try:
            partitions.append(
                Partition(
                    layer_name,
                    "classification_layers",
                    dsm,
                    ref,
                    output_dir,
                    geo_ref=geo_ref,
                    dec_ref_path=cfg["inputRef"]["path"],
                    dec_dem_path=cfg["inputDSM"]["path"],
                    init_disp_x=cfg["plani_opts"]["disp_init"]["x"],
                    init_disp_y=cfg["plani_opts"]["disp_init"]["y"],
                    dx=cfg["plani_results"]["dx"]["nuth_offset"],
                    dy=cfg["plani_results"]["dy"]["nuth_offset"],
                    **clayer
                )
            )
        except Exception as error:
            traceback.print_exc()
            print(
                (
                    "Cannot create partition for {}:{} -> {}".format(
                        layer_name, clayer, error
                    )
                )
            )

    # Create the fusion partition
    if len(partitions) > 1:
        try:
            partitions.append(
                FusionPartition(partitions, output_dir, geo_ref=geo_ref)
            )
        except NotEnoughDataToPartitionError:
            logging.info("Partitions could ne be created")

    if len(partitions) == 0:
        try:
            clayer = {}
            partitions.append(
                Partition(
                    "global",
                    "classification_layers",
                    dsm,
                    ref,
                    output_dir,
                    geo_ref=geo_ref,
                    **clayer
                )
            )
        except NotEnoughDataToPartitionError:
            logging.info("Partitions could not be created")

    for p in partitions:
        logging.debug("list of already classification layers: {}".format(p))

    return partitions


def alti_diff_stats(
    cfg: Dict,
    dsm: xr.Dataset,
    ref: xr.Dataset,
    alti_map: xr.Dataset,
    display: bool = False,
    remove_outliers: bool = False,
    geo_ref: bool = True,
):
    """
    Computes alti error stats with graphics and tables support.

    If cfg['stats_opt']['class_type'] is not None,
    those stats can be partitioned into different sets.
    The sets are radiometric ranges used to classify a support image.
    May the support image be the slope image associated with the reference DSM
    then the sets are slopes ranges
    and the stats are provided by classes of slopes ranges.

    Actually, if cfg['stats_opt']['class_type'] is 'slope'
    then computeStats first computes slope image and classify stats over slopes.

    If cfg['stats_opt']['class_type'] is 'user' then a user support image
    must be given to be classified over cfg['stats_opt']['class_rad_range']
    intervals so it can partitioned the stats.

    When cfg['stats_opt']['class_type']['class_coherent'] is set to True
    then two images to classify are required
    (one associated with the reference DEM and one with the other one).
    The results will be presented through 3 modes:

    standard mode,

    coherent mode, where only alti errors
    values associated with coherent classes
    between both classified images are used

    incoherent mode (the coherent complementary one).

    :param cfg: config file
    :type cfg: Dict
    :param dsm: dsm
    :type dsm: xr.Dataset
    :param ref: xarray Dataset, coregistered ref
    :type ref: xr.Dataset
    :param alti_map: xarray Dataset, dsm - ref
    :type alti_map: xr.Datast
    :param display: boolean, display option
            (set to False to save plot on file system)
    :type display: bool
    :param remove_outliers: boolean, set to True to
            remove outliers ( x < mu - 3sigma ; x > mu + 3sigma)
    :type remove_outliers: bool
    :param geo_ref: boolean, set to False if images are not georeferenced
    :type geo_ref: bool
    :return:
    """

    def get_title(cfg):
        """Create title for alti_diff_stats"""
        if geo_ref:
            # Set future plot title with bias and % of nan values as part of it
            title = ["DEM quality performance"]
            dx = cfg["plani_results"]["dx"]
            dy = cfg["plani_results"]["dy"]
            biases = {
                "dx": {
                    "value_m": dx["bias_value"],
                    "value_p": dx["bias_value"] / ref.attrs["xres"],
                },
                "dy": {
                    "value_m": dy["bias_value"],
                    "value_p": dy["bias_value"] / ref.attrs["yres"],
                },
            }
            title.append(
                "(mean biases : "
                "dx : {:.2f}m (roughly {:.2f}pixel); "
                "dy : {:.2f}m (roughly {:.2f}pixel);)".format(
                    biases["dx"]["value_m"],
                    biases["dx"]["value_p"],
                    biases["dy"]["value_m"],
                    biases["dy"]["value_p"],
                )
            )
            rect_ref_cfg = cfg["alti_results"]["rectifiedRef"]
            rect_dsm_cfg = cfg["alti_results"]["rectifiedDSM"]
            title.append(
                "(holes or no data stats: "
                "Reference DSM  % nan values : {:.2f}%; "
                "DSM to compare % nan values : {:.2f}%;)".format(
                    100
                    * (
                        1
                        - float(rect_ref_cfg["nb_valid_points"])
                        / float(rect_ref_cfg["nb_points"])
                    ),
                    100
                    * (
                        1
                        - float(rect_dsm_cfg["nb_valid_points"])
                        / float(rect_dsm_cfg["nb_points"])
                    ),
                )
            )
        else:
            title = "title"
        return title

    def get_thresholds_in_meters(cfg):
        """
        Create list of threshold in meters.
        """
        # If required, get list of altitude thresholds and adjust the unit
        list_threshold_m = None
        if cfg["stats_opts"]["elevation_thresholds"]["list"]:
            # Convert thresholds to meter
            # since all DEMcompare elevation unit is "meter"
            original_unit = cfg["stats_opts"]["elevation_thresholds"]["zunit"]
            list_threshold_m = [
                ((threshold * u.Unit(original_unit)).to(u.meter)).value
                for threshold in cfg["stats_opts"]["elevation_thresholds"][
                    "list"
                ]
            ]
        return list_threshold_m

    # Get outliers free mask (array of True where value is no outlier)
    if remove_outliers:
        outliers_free_mask = get_outliers_free_mask(
            alti_map["im"].data, alti_map.attrs["no_data"]
        )
    else:
        outliers_free_mask = 1

    # There can be multiple ways to partition the stats.
    # We gather them all inside a list here:
    partitions = create_partitions(dsm, ref, cfg, geo_ref=geo_ref)

    # For every partition get stats and save them as plots and tables
    cfg["stats_results"]["partitions"] = {}
    for p in partitions:
        # Compute stats for each mode and every sets
        mode_stats, mode_masks, mode_names = get_stats_per_mode(
            alti_map,
            sets_masks=p.sets_masks,
            sets_labels=p.sets_labels,
            sets_names=p.sets_names,
            elevation_thresholds=get_thresholds_in_meters(cfg),
            outliers_free_mask=outliers_free_mask,
        )

        # Save stats as plots, csv and json and do so for each mode
        p.stats_mode_json = save_as_graphs_and_tables(
            alti_map["im"].data,
            p.stats_dir,
            p.plots_dir,
            p.histograms_dir,
            [mode_mask * outliers_free_mask for mode_mask in mode_masks],
            mode_names,
            mode_stats,
            p.sets_masks[0],  # do not need 'ref' and 'dsm' only one of them
            p.sets_labels,
            p.sets_colors,
            plot_title="\n".join(get_title(cfg)),
            bin_step=cfg["stats_opts"]["alti_error_threshold"]["value"],
            display=display,
            plot_real_hist=cfg["stats_opts"]["plot_real_hists"],
            geo_ref=geo_ref,
        )

        # get partition stats results
        cfg["stats_results"]["partitions"][p.name] = p.stats_results

        # TODO two possibilities :
        # - call generate_report() itself directly
        # - Best way: the partition generate itself its HTML page
        #   (string to create) and generate_report() concatenate generated pages


def save_as_graphs_and_tables(
    data_array: np.ndarray,
    stats_dir: str,
    outplotdir: str,
    outhistdir: str,
    mode_masks: List,
    mode_names: List[str],
    mode_stats,
    sets_masks,
    sets_labels,
    sets_colors,
    plot_title: str = "Title",
    bin_step: float = 0.1,
    display: bool = False,
    plot_real_hist: bool = True,
    geo_ref: bool = True,
):
    """

    :param data_array:
    :type data_array: np.ndarray
    :param out_dir:
    :type out_dir: str
    :param mode_masks:
    :type mode_masks:
    :param mode_names: List[str]
    :type mode_names:
    :param mode_stats:
    :type mode_stats:
    :param sets_masks:
    :type sets_masks:
    :param sets_labels:
    :type sets_labels:
    :param sets_colors:
    :type sets_colors:
    :param plot_title:
    :type plot_title: str
    :param bin_step:
    :type bin_step: float
    :param display:
    :type display: bool
    :param plot_real_hist:
    :type plot_real_hist: bool
    :param geo_ref: boolean, set to False if images are not georeferenced
    :type geo_ref:
    :return:
    """
    mode_output_json_files = {}
    # TODO (peut etre prevoir une activation optionnelle du plotage...)
    if sets_labels is not None:
        sets_labels = ["all"] + sets_labels
    if sets_colors is not None:
        sets_colors = np.array([(0, 0, 0)] + list(sets_colors))
    else:
        sets_colors = np.array([(0, 0, 0)])

    for mode_idx, mode_name_item in enumerate(mode_names):
        #
        # Create plots for the actual mode and for all sets
        #
        # -> we are then ready to do some plots !
        if geo_ref:
            try:
                plot_files, labels, colors = plot_histograms(
                    data_array,
                    bin_step=bin_step,
                    to_keep_mask=mode_masks[mode_idx],
                    sets=[np.ones(data_array.shape, dtype=bool)] + sets_masks,
                    sets_labels=sets_labels,
                    sets_colors=sets_colors,
                    plot_title=plot_title,
                    outplotdir=outplotdir,
                    outhistdir=outhistdir,
                    save_prefix=mode_name_item,
                    display=display,
                    plot_real_hist=plot_real_hist,
                )
            except NoPointsToPlot:
                print(("Nothing to plot for mode {} ".format(mode_name_item)))
                continue

        #
        # Save results as .json and .csv file
        #
        mode_output_json_files[mode_name_item] = os.path.join(
            stats_dir, "stats_results_" + mode_name_item + ".json"
        )
        if geo_ref:
            save_results(
                mode_output_json_files[mode_name_item],
                mode_stats[mode_idx],
                labels_plotted=labels,
                plot_files=plot_files,
                plot_colors=colors,
                to_csv=True,
            )
        else:
            save_results(
                mode_output_json_files[mode_name_item],
                mode_stats[mode_idx],
                to_csv=True,
            )

    return mode_output_json_files


def get_stats_per_mode(
    data: np.ndarray,
    sets_masks: List = None,
    sets_labels: List = None,
    sets_names: List = None,
    elevation_thresholds: List[float] = None,
    outliers_free_mask=None,
) -> Tuple[List, List, List]:
    """
    Generates alti error stats with graphics and csv tables.

    Stats are computed based on support images which can be viewed
    as classification layers. The layers are represented by the support_sets
    which partitioned the alti_map indices.
    Hence stats are computed on each set separately.

    There can be one or two support_imgs, associated to just the same amount
    of supports_sets. Both arguments being lists.
    If two such classification layers are given,
    then this method also produces stats based on 3 modes:

    standard mode

    coherent mode: where only alti errors values associated
    with coherent classes between both classified images are used

    incoherent mode: the coherent complementary one

    :param data: array to compute stats from
    :type data: np.ndarray
    :param sets_masks: [] of one or two array
            (sets partitioning the support_img) of size equal to data ones
    :type sets_masks: List
    :param sets_labels: sets labels
    :type sets_labels: List
    :param sets_names: sets names
    :type sets_names: List
    :param elevation_thresholds: list of elevation thresholds
    :type elevation_thresholds: List[float]
    :param outliers_free_mask:
    :type outliers_free_mask:
    :return: stats, masks, names per mode
    :rtype: List, List List
    """

    # Get mode masks and names
    # (sets_masks will be cross checked if len(sets_masks)==2)
    mode_masks, mode_names = create_mode_masks(data, sets_masks)

    # Next is done for all modes
    mode_stats = []
    for mode_idx, _ in enumerate(mode_names):

        # Compute stats for all sets of a single mode
        mode_stats.append(
            get_stats(
                data["im"].data,
                to_keep_mask=mode_masks[mode_idx],
                outliers_free_mask=outliers_free_mask,
                sets=sets_masks[
                    0
                ],  # do not need ref and dsm but only one of them
                sets_labels=sets_labels,
                sets_names=sets_names,
                list_threshold=elevation_thresholds,
            )
        )

    return mode_stats, mode_masks, mode_names


def wave_detection(cfg: Dict, dh):
    """
    Detect potential oscillations inside dh

    :param cfg: config file
    :type cfg: Dict
    :param dh: dsm - ref
    :type dh: xr.Dataset
    :return: None
    """
    # Compute mean dh row and mean dh col
    # -> then compute min between dh mean row (col) vector and dh rows (cols)
    res = {
        "row_wise": np.zeros(dh["im"].data.shape, dtype=np.float32),
        "col_wise": np.zeros(dh["im"].data.shape, dtype=np.float32),
    }
    axis = -1
    for dim in list(res.keys()):
        axis += 1
        mean = np.nanmean(dh["im"].data, axis=axis)
        if axis == 1:
            # for axis == 1, we need to transpose the array to substitute it
            # to dh.r otherwise 1D array stays row array
            mean = np.transpose(
                np.ones((1, mean.size), dtype=np.float32) * mean
            )
        res[dim] = dh["im"].data - mean

        cfg["stats_results"]["images"]["list"].append(dim)
        cfg["stats_results"]["images"][dim] = copy.deepcopy(
            cfg["alti_results"]["dzMap"]
        )
        cfg["stats_results"]["images"][dim].pop("nb_points")
        cfg["stats_results"]["images"][dim]["path"] = os.path.join(
            cfg["outputDir"],
            get_out_file_path("dh_{}_wave_detection.tif".format(dim)),
        )

        georaster = read_img_from_array(
            res[dim], from_dataset=dh, no_data=-32768
        )
        save_tif(georaster, cfg["stats_results"]["images"][dim]["path"])
