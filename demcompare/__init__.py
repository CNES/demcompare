#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022 Centre National d'Etudes Spatiales (CNES).
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
# pylint:disable=too-many-branches
"""
Demcompare init module file.
Demcompare aims at coregistering and comparing two Digital Elevation Models(DEM)
"""

# Standard imports
import logging
import logging.config
import os
from importlib.metadata import version
from typing import List, Tuple, Union

# Third party imports
import xarray as xr

# Demcompare imports
from . import log_conf, report
from .coregistration import Coregistration
from .dem_processing import DemProcessing
from .dem_tools import (
    compute_and_save_image_plots,
    compute_dem_slope,
    load_dem,
    reproject_dems,
    save_dem,
    verify_fusion_layers,
)
from .helpers_init import (
    compute_initialization,
    get_output_files_paths,
    save_config_file,
)
from .internal_typing import ConfigType
from .stats_dataset import StatsDataset
from .stats_processing import StatsProcessing

# VERSION through setuptools_scm when python3 > 3.8
try:
    __version__ = version("demcompare")
except Exception:  # pylint: disable=broad-except
    __version__ = "unknown"

__author__ = "CNES"
__email__ = "cars@cnes.fr"


def run(
    json_file_path: str,
    loglevel: int = logging.WARNING,
):
    """
    Demcompare RUN execution.

    :param json_file_path: Input Json configuration file
    :type json_file_path: str
    :param loglevel: Choose Loglevel (default: WARNING)
    :type loglevel: int
    """

    # Initialization

    # Get cfg from json file, checking inputs
    cfg = compute_initialization(json_file_path)

    # Create output_dir from updated absolute path
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Logging configuration
    log_conf.setup_logging(default_level=loglevel)
    # Add output logging file
    log_conf.add_log_file(cfg["output_dir"])

    logging.info("*** Demcompare ***")
    logging.info("Output directory: %s", cfg["output_dir"])

    # Save initial config
    # with inputs absolute paths into output_dir
    save_config_file(
        os.path.join(cfg["output_dir"], os.path.basename(json_file_path)), cfg
    )
    logging.debug("Demcompare configuration: %s", cfg)

    # Get input ref and input sec dem datasets from cfg
    input_ref, input_sec = load_input_dems(cfg)

    logging.info("Input Reference DEM (REF): %s", input_ref.input_img)
    # if two references, show input secondary DEM
    if input_sec:
        logging.info("Input Secondary DEM (SEC): %s", input_sec.input_img)

    # Create list of datasets
    stats_datasets: List[StatsDataset] = []

    # If coregistration step is present
    if "coregistration" in cfg:
        # update cfg coregistration output (created in coreg sub pipeline)
        cfg["coregistration"]["output_dir"] = os.path.join(
            cfg["output_dir"], "coregistration"
        )
        # Do coregistration and obtain initial
        # and final intermediate dems for stats computation
        (
            input_stats_sec,
            input_stats_ref,
        ) = run_coregistration(cfg["coregistration"], input_ref, input_sec)

    else:
        # If input_sec is not None, reproject DEMs
        if input_sec:
            input_stats_sec, input_stats_ref, _ = reproject_dems(
                input_sec,
                input_ref,
                sampling_source=(
                    cfg["sampling_source"] if "sampling_source" in cfg else None
                ),
            )
        # If input_sec is None, input_stats_sec=None
        else:
            input_stats_ref, input_stats_sec = input_ref, None

    # If only stats is present
    if "statistics" in cfg:
        logging.info("[Stats]")
        logging.info("Altimetric stats generation")

        # Loop over the DEM processing methods in cfg["statistics"]
        for dem_processing_method in cfg["statistics"]:
            # create directory for dem processing method stats
            os.makedirs(
                cfg["statistics"][dem_processing_method]["output_dir"],
                exist_ok=True,
            )

            # Create a DEM processing object for each DEM processing method
            dem_processing_object = DemProcessing(dem_processing_method)

            # Obtain output paths for initial dem diff without coreg
            (
                dem_path,
                plot_file_path,
                plot_path_cdf,
                csv_path_cdf,
                plot_path_pdf,
                csv_path_pdf,
                plot_path_svf,
                plot_path_hillshade,
            ) = get_output_files_paths(
                cfg["output_dir"], dem_processing_method, "dem_for_stats"
            )

            # Compute slope and add it as a classification_layer
            # in case a classification of type slope is required
            # The ref is considered the main classification,
            # the slope of the sec dem will be used for the
            # intersection-exclusion
            input_stats_ref = compute_dem_slope(input_stats_ref)
            if input_stats_sec:
                input_stats_sec = compute_dem_slope(input_stats_sec)

            # If defined, verify fusion layers according to the cfg
            if (
                "classification_layer"
                in cfg["statistics"][dem_processing_method]
            ):
                if (
                    "fusion"
                    in cfg["statistics"][dem_processing_method][
                        "classification_layers"
                    ]
                ):
                    verify_fusion_layers(
                        input_stats_ref,
                        cfg["statistics"][dem_processing_method][
                            "classification_layers"
                        ],
                        support="ref",
                    )
                    if input_stats_sec:
                        verify_fusion_layers(
                            input_stats_sec,
                            cfg["statistics"][dem_processing_method][
                                "classification_layers"
                            ],
                            support="sec",
                        )
            logging.info(" Dem processing: %s ", dem_processing_object.type)
            stats_dem = dem_processing_object.process_dem(
                input_stats_ref, input_stats_sec
            )

            # Save stats_dem for two states
            save_dem(stats_dem, dem_path)

            # Compute and save initial altitude diff image plots
            compute_and_save_image_plots(
                stats_dem,
                plot_file_path,
                fig_title=dem_processing_object.fig_title,
                colorbar_title=dem_processing_object.colorbar_title,
                cmap=dem_processing_object.cmap,
                vmin_plot=(
                    cfg["statistics"][dem_processing_method]["vmin_plot"]
                    if "vmin_plot" in cfg["statistics"][dem_processing_method]
                    else None
                ),
                vmax_plot=(
                    cfg["statistics"][dem_processing_method]["vmax_plot"]
                    if "vmax_plot" in cfg["statistics"][dem_processing_method]
                    else None
                ),
            )

            # Create StatsComputation object
            stats_processing = StatsProcessing(
                cfg["statistics"][dem_processing_method],
                stats_dem,
                dem_processing_method=dem_processing_method,
            )

            # For the initial_dh, compute cdf and pdf stats
            # on the global classification layer only (diff, pdf, cdf)
            plot_metrics = [
                {
                    "cdf": {
                        "remove_outliers": cfg["statistics"][
                            dem_processing_method
                        ]["remove_outliers"],
                        "output_plot_path": plot_path_cdf,
                        "output_csv_path": csv_path_cdf,
                    }
                },
                {
                    "pdf": {
                        "remove_outliers": cfg["statistics"][
                            dem_processing_method
                        ]["remove_outliers"],
                        "output_plot_path": plot_path_pdf,
                        "output_csv_path": csv_path_pdf,
                    }
                },
                {
                    "svf": {
                        "remove_outliers": cfg["statistics"][
                            dem_processing_method
                        ]["remove_outliers"],
                        "plot_path": plot_path_svf,
                    }
                },
                {
                    "hillshade": {
                        "remove_outliers": cfg["statistics"][
                            dem_processing_method
                        ]["remove_outliers"],
                        "plot_path": plot_path_hillshade,
                    }
                },
            ]

            # generate intermediate stats results CDF and PDF for report
            # refacto type hinting standardize metrics input type
            stats_processing.compute_stats(
                classification_layer=["global"],
                metrics=plot_metrics,  # type: ignore
            )

            # Compute stats according to the input stats configuration
            stats_dataset = stats_processing.compute_stats()

            stats_datasets.append(stats_dataset)

    # Save full final config
    # with inputs absolute paths into output_dir
    save_config_file(os.path.join(cfg["output_dir"], "full_config.json"), cfg)

    # Generate report if statistics are computed (stats_dataset defined)
    # and report configuration is set
    if "report" in cfg and "statistics" in cfg:
        logging.info("[Report]")
        report.generate_report(
            cfg=cfg,
            stats_datasets=stats_datasets,
        )


def load_input_dems(
    cfg: ConfigType,
) -> Tuple[xr.Dataset, Union[None, xr.Dataset]]:
    """
    Loads the input dems according to the input cfg

    :param cfg: input configuration
    :type cfg: ConfigType
    :return: input_ref and input_dem datasets or None
    :rtype:   Tuple(xr.Dataset, xr.Dataset)
          The xr.Datasets containing :

          - im : 2D (row, col) xarray.DataArray float32
          - trans: 1D (trans_len) xarray.DataArray
    """
    # Create input datasets
    ref = load_dem(
        cfg["input_ref"]["path"],
        nodata=(
            cfg["input_ref"]["nodata"] if "nodata" in cfg["input_ref"] else None
        ),
        geoid_georef=(
            cfg["input_ref"]["geoid_georef"]
            if "geoid_georef" in cfg["input_ref"]
            else False
        ),
        geoid_path=(
            cfg["input_ref"]["geoid_path"]
            if "geoid_path" in cfg["input_ref"]
            else None
        ),
        zunit=(
            cfg["input_ref"]["zunit"] if "zunit" in cfg["input_ref"] else "m"
        ),
        input_roi=(
            cfg["input_ref"]["roi"] if "roi" in cfg["input_ref"] else False
        ),
        classification_layers=(
            cfg["input_ref"]["classification_layers"]
            if "classification_layers" in cfg["input_ref"]
            else None
        ),
    )
    if "input_sec" in cfg:
        sec = load_dem(
            cfg["input_sec"]["path"],
            nodata=(
                cfg["input_sec"]["nodata"]
                if "nodata" in cfg["input_sec"]
                else None
            ),
            geoid_georef=(
                cfg["input_sec"]["geoid_georef"]
                if "geoid_georef" in cfg["input_sec"]
                else False
            ),
            geoid_path=(
                cfg["input_sec"]["geoid_path"]
                if "geoid_path" in cfg["input_sec"]
                else None
            ),
            zunit=(
                cfg["input_sec"]["zunit"]
                if "zunit" in cfg["input_sec"]
                else "m"
            ),
            input_roi=(
                cfg["input_sec"]["roi"] if "roi" in cfg["input_sec"] else False
            ),
            classification_layers=(
                cfg["input_sec"]["classification_layers"]
                if "classification_layers" in cfg["input_sec"]
                else None
            ),
        )

    else:
        sec = None

    return ref, sec


def run_coregistration(
    cfg: ConfigType, input_ref: xr.Dataset, input_sec: xr.Dataset
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Runs the dems coregistration

    :param cfg: coregistration configuration
    :type cfg: ConfigType
    :param input_ref: input ref
    :type input_ref: xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :param input_sec: input dem
    :type input_sec: xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :return: reproj_coreg_sec, reproj_coreg_ref
    :rtype:   Tuple(xr.Dataset, xr.Dataset)
             The xr.Datasets containing :

             - im : 2D (row, col) xarray.DataArray float32
             - trans: 1D (trans_len) xarray.DataArray
    """
    logging.info("[Coregistration]")
    # Create coregistration object
    coregistration_ = Coregistration(cfg)

    # Compute coregistration
    _ = coregistration_.compute_coregistration(input_sec, input_ref)

    # Get coregistration_results dict
    coregistration_results = coregistration_.coregistration_results

    # Save coregistration_results
    save_config_file(
        os.path.join(
            cfg["output_dir"],
            "./coregistration_results.json",
        ),
        coregistration_results,
    )

    # Get internal dems
    reproj_coreg_ref = coregistration_.reproj_coreg_ref
    reproj_coreg_sec = coregistration_.reproj_coreg_sec

    return (
        reproj_coreg_sec,
        reproj_coreg_ref,
    )
