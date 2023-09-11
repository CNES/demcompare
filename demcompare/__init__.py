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
"""
Demcompare init module file.
Demcompare aims at coregistering and comparing two Digital Elevation Models(DEM)
"""
import copy

# Standard imports
import logging
import logging.config
import os
from importlib.metadata import version
from typing import List, Tuple, Union

# Third party imports
import xarray as xr

# Demcompare imports
from . import helpers_init, log_conf, report
from .coregistration import Coregistration
from .dem_processing import DemProcessing
from .dem_representation import DemRepresentation
from .dem_tools import (
    compute_dem_slope,
    load_dem,
    reproject_dems,
    save_dem,
    verify_fusion_layers,
)
from .internal_typing import ConfigType
from .output_tree_design import get_out_file_path
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
    :type json_file: str
    :param loglevel: Choose Loglevel (default: WARNING)
    :type loglevel: int
    """

    # Initialization

    # Configuration copy, checking inputs, create output dir tree
    cfg = helpers_init.compute_initialization(json_file_path)

    # Logging configuration
    log_conf.setup_logging(default_level=loglevel)
    # Add output logging file
    log_conf.add_log_file(cfg["output_dir"])

    logging.info("*** Demcompare ***")
    logging.info("Output directory: %s", cfg["output_dir"])
    logging.debug("Demcompare configuration: %s", cfg)

    input_ref, input_sec = load_input_dems(cfg)

    logging.info("Input Reference DEM (REF): %s", input_ref.input_img)
    # if two references, show input secondary DEM
    if input_sec:
        logging.info("Input Secondary DEM (SEC): %s", input_sec.input_img)

    # Create list of datasets
    stats_datasets: List[StatsDataset] = []

    # If coregistration step is present
    if "coregistration" in cfg:
        # Do coregistration and obtain initial
        # and final intermediate dems for stats computation
        (
            reproj_coreg_sec,
            reproj_coreg_ref,
        ) = run_coregistration(cfg["coregistration"], input_ref, input_sec)

        if "statistics" in cfg:
            # Compute stats after coregistration
            stats_datasets = compute_stats_after_coregistration(
                cfg["statistics"],
                reproj_coreg_sec,
                reproj_coreg_ref,
            )

    # If only stats is present
    elif "statistics" in cfg:
        logging.info("[Stats]")
        # If both dems have been defined, compute altitude difference for stats
        if input_ref and input_sec:
            logging.info("(REF-SEC) altimetric stats generation")

            # Loop over the DEM processing methods in cfg["statistics"]
            for dem_processing_method in cfg["statistics"]:
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
                ) = helpers_init.get_output_files_paths(
                    cfg["output_dir"],
                    dem_processing_method,
                    file_name="dem_for_stats",
                )

                reproj_sec, reproj_ref, _ = reproject_dems(
                    input_sec,
                    input_ref,
                    sampling_source=cfg["statistics"][dem_processing_method][
                        "sampling_source"
                    ]
                    if "sampling_source"
                    in cfg["statistics"][dem_processing_method]
                    else None,
                )

                # Compute slope and add it as a classification_layer
                # in case a classification of type slope is required
                # The ref is considered the main classification,
                # the slope of the sec dem will be used for the
                # intersection-exclusion
                reproj_ref = compute_dem_slope(reproj_ref)
                reproj_sec = compute_dem_slope(reproj_sec)

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
                            reproj_ref,
                            cfg["statistics"][dem_processing_method][
                                "classification_layers"
                            ],
                            support="ref",
                        )
                        verify_fusion_layers(
                            reproj_sec,
                            cfg["statistics"][dem_processing_method][
                                "classification_layers"
                            ],
                            support="sec",
                        )

                stats_dem = dem_processing_object.process_dem(
                    reproj_ref, reproj_sec
                )

                # Save stats_dem for two states
                save_dem(stats_dem, dem_path)

                dem_representation_object = DemRepresentation("dem")

                # Compute and save initial altitude diff image plots
                dem_representation_object.compute_and_save_image_plots(
                    stats_dem,
                    plot_file_path,
                    fig_title=dem_processing_object.fig_title,
                    colorbar_title=dem_processing_object.colorbar_title,
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
                ]

                # generate intermediate stats results CDF and PDF for report
                stats_processing.compute_stats(
                    classification_layer=["global"],
                    metrics=plot_metrics,  # type: ignore
                )

                # Compute stats according to the input stats configuration
                stats_dataset = stats_processing.compute_stats()

                stats_datasets.append(stats_dataset)

        else:
            # only one dem
            logging.info("(REF) altimetric stats generation")

            # Loop over the DEM processing methods in cfg["statistics"]
            for dem_processing_method in cfg["statistics"]:
                # Warning: "ref-curvature" does not work with "Slope0" as "classification_layers" # noqa: E501, B950 # pylint: disable=line-too-long

                # Compute slope and add it as a classification_layer
                # in case a classification of type slope is required
                input_ref = compute_dem_slope(input_ref)
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
                            input_ref,
                            cfg["statistics"][dem_processing_method][
                                "classification_layers"
                            ],
                            support="ref",
                        )

                # Create a DEM processing object for each DEM processing method
                dem_processing_object = DemProcessing(dem_processing_method)

                # Define stats_dem as the input dem
                stats_dem = dem_processing_object.process_dem(input_ref)
                if "representation" in cfg["statistics"][dem_processing_method]:
                    for dem_representation_method in cfg["statistics"][
                        dem_processing_method
                    ]["representation"]:
                        # Obtain output paths for initial dem diff without coreg
                        (
                            dem_path,
                            plot_file_path,
                            plot_path_cdf,
                            csv_path_cdf,
                            plot_path_pdf,
                            csv_path_pdf,
                        ) = helpers_init.get_output_files_paths(
                            cfg["output_dir"],
                            dem_processing_method,
                            dem_representation_method,
                            "dem_for_stats",
                        )

                        dem_representation_object = DemRepresentation(
                            dem_representation_method,
                            cfg["statistics"][dem_processing_method][
                                "representation"
                            ][dem_representation_method],
                        )

                        # Compute and save initial altitude diff image plots
                        dem_representation_object.compute_and_save_image_plots(
                            stats_dem,
                            plot_file_path,
                            fig_title=dem_representation_object.fig_title,
                            colorbar_title=dem_representation_object.colorbar_title,  # noqa: E501, B950 # pylint: disable=line-too-long
                        )

                else:
                    dem_representation_object = DemRepresentation("dem")

                    (
                        dem_path,
                        plot_file_path,
                        plot_path_cdf,
                        csv_path_cdf,
                        plot_path_pdf,
                        csv_path_pdf,
                    ) = helpers_init.get_output_files_paths(
                        cfg["output_dir"],
                        dem_processing_method,
                        "dem_for_stats",
                    )

                    # Compute and save initial altitude diff image plots
                    dem_representation_object.compute_and_save_image_plots(
                        stats_dem,
                        plot_file_path,
                        fig_title=dem_processing_object.fig_title,
                        colorbar_title=dem_processing_object.colorbar_title,
                    )

                # Save stats_dem for two states
                save_dem(stats_dem, dem_path)

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
                ]

                # generate intermediate stats results CDF and PDF for report
                stats_processing.compute_stats(
                    classification_layer=["global"],
                    metrics=plot_metrics,  # type: ignore
                )

                # Compute stats according to the input stats configuration
                stats_dataset = stats_processing.compute_stats()

                stats_datasets.append(stats_dataset)

    # Generate report if statistics are computed (stats_dataset defined)
    # and report configuration is set
    if "report" in cfg and "statistics" in cfg:
        logging.info("[Report]")
        report.generate_report(
            cfg=cfg,
            stats_dataset=stats_dataset,
        )


def load_input_dems(
    cfg: ConfigType,
) -> Tuple[xr.Dataset, Union[None, xr.Dataset]]:
    """
    Loads the input dems according to the input cfg

    :param cfg: input configuration
    :type cfg: ConfigType
    :return: input_ref and input_dem datasets or None
    :rtype:   Tuple(xr.Dataset, xr.dataset)
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
    :rtype:   Tuple(xr.Dataset, xr.dataset)
             The xr.Datasets containing :

             - im : 2D (row, col) xarray.DataArray float32
             - trans: 1D (trans_len) xarray.DataArray
    """
    logging.info("[Coregistration]")
    # Create coregistration object
    coregistration_ = Coregistration(cfg)
    # Compute coregistration to get applicable transformation object
    transformation = coregistration_.compute_coregistration(
        input_sec, input_ref
    )
    # Apply coregistration offsets to the original DEM and store it
    # reprojection is also done.
    coreg_sec = transformation.apply_transform(input_sec)
    # Get demcompare_results dict
    demcompare_results = coregistration_.demcompare_results

    # Save the coregistered DEM (even without save_optional_outputs option)
    # - coreg_SEC.tif -> coregistered sec
    save_dem(
        coreg_sec,
        os.path.join(cfg["output_dir"], get_out_file_path("coreg_SEC.tif")),
    )
    # Save demcompare_results
    helpers_init.save_config_file(
        os.path.join(
            cfg["output_dir"],
            get_out_file_path("demcompare_results.json"),
        ),
        demcompare_results,
    )

    # Get internal dems
    reproj_coreg_ref = coregistration_.reproj_coreg_ref
    reproj_coreg_sec = coregistration_.reproj_coreg_sec

    return (
        reproj_coreg_sec,
        reproj_coreg_ref,
    )


def compute_stats_after_coregistration(
    cfg: ConfigType,
    coreg_sec: xr.Dataset,
    coreg_ref: xr.Dataset,
) -> List[StatsDataset]:
    """
    Compute stats after coregistration

    For the initial_dh and final_dh the alti_diff plot, the
    cdf and the pdf are computed if the output_dir has
    been specified in order to evaluate the
    coregistration effect.

    For the final_dh, the different classification layers
    and metrics specified in the input cfg are also computed.

    :param cfg: configuration dictionary
    :type cfg: ConfigType
    :param coreg_sec: coreg dem to align xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type coreg_sec: xr.Dataset
    :param coreg_ref: coreg reference dem xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type coreg_ref: xr.Dataset
    :param initial_sec: optional initial dem
                to align xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type initial_ref: xr.Dataset
    :param initial_ref: optional initial reference dem xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type initial_ref: xr.Dataset
    :return: List[StatsDataset]
    :rtype: List[StatsDataset]
    """
    logging.info("[Stats]")
    logging.info("(COREG_REF-COREG_SEC) altimetric stats generation")

    # Compute slope and add it as a classification_layer in case
    # a classification of type slope is required
    # The ref is considered the main classification,
    # the slope of the sec dem will be used for the intersection-exclusion
    coreg_ref = compute_dem_slope(coreg_ref)
    coreg_sec = compute_dem_slope(coreg_sec)

    stats_datasets: List[StatsDataset] = []

    # Loop over the DEM processing methods in cfg
    for dem_processing_method in cfg:
        # Create a DEM processing object for each DEM processing method
        dem_processing_object = DemProcessing(dem_processing_method)

        # Take the part of the config associated with the DEM processing method
        cfg_method = cfg[dem_processing_method]

        # If defined, verify fusion layers according to the cfg
        if "classification_layer_masks" in cfg_method:
            if "fusion" in cfg_method["classification_layers"]:
                verify_fusion_layers(
                    coreg_ref,
                    cfg_method["classification_layers"],
                    support="ref",
                )
                verify_fusion_layers(
                    coreg_sec,
                    cfg_method["classification_layers"],
                    support="sec",
                )

        # Compute altitude diff
        final_altitude_diff = dem_processing_object.process_dem(
            coreg_ref, coreg_sec
        )

        # Create StatsComputation object for the final_dh
        final_stats_cfg = copy.deepcopy(cfg_method)
        stats_processing_final = StatsProcessing(
            final_stats_cfg,
            final_altitude_diff,
            input_diff=True,
            dem_processing_method=dem_processing_method,
        )

        # Obtain output paths
        (
            dem_path,
            plot_file_path,
            plot_path_cdf,
            csv_path_cdf,
            plot_path_pdf,
            csv_path_pdf,
        ) = helpers_init.get_output_files_paths(
            cfg_method["output_dir"],
            dem_processing_method,
            file_name="dem_for_stats",
        )

        # Save final altitude diff
        save_dem(final_altitude_diff, dem_path)

        dem_representation_object = DemRepresentation("dem")

        # Compute and save final altitude diff image plots
        dem_representation_object.compute_and_save_image_plots(
            dem=final_altitude_diff,
            plot_path=plot_file_path,
            fig_title=dem_processing_object.fig_title,
            colorbar_title=dem_processing_object.colorbar_title,
        )

        # For the final_dh, first compute plot stats on the
        # global classification layer only (diff, pdf, cdf)

        plot_metrics = [
            {
                "cdf": {
                    "remove_outliers": cfg_method["remove_outliers"],
                    "output_plot_path": plot_path_cdf,
                    "output_csv_path": csv_path_cdf,
                }
            },
            {
                "pdf": {
                    "remove_outliers": cfg_method["remove_outliers"],
                    "output_plot_path": plot_path_pdf,
                    "output_csv_path": csv_path_pdf,
                }
            },
        ]

        # Generate intermediate stats to compare pdf and cdf before and after
        stats_processing_final.compute_stats(
            classification_layer=["global"],
            metrics=plot_metrics,  # type: ignore
        )

        # For the final_dh, also compute all classif layer default metric stats
        stats_dataset = stats_processing_final.compute_stats()

        stats_datasets.append(stats_dataset)

    return stats_datasets
