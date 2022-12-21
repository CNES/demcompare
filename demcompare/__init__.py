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
from typing import Dict, Tuple, Union

# Third party imports
import xarray as xr

# Demcompare imports
from . import helpers_init, log_conf, report
from .coregistration import Coregistration
from .dem_tools import (
    compute_alti_diff_for_stats,
    compute_and_save_image_plots,
    compute_dem_slope,
    compute_dems_diff,
    load_dem,
    reproject_dems,
    save_dem,
    verify_fusion_layers,
)
from .output_tree_design import get_out_dir, get_out_file_path
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
    json_file: str,
    loglevel=logging.WARNING,
):
    """
    Demcompare RUN execution.

    :param json_file: Input Json configuration file
    :type json_file: str
    :param loglevel: Choose Loglevel (default: WARNING)
    :type loglevel: logging.WARNING
    """

    # Initialization
    # Configuration copy, checking inputs, create output dir tree
    cfg = helpers_init.compute_initialization(json_file)
    log_conf.setup_logging(default_level=loglevel)
    # Add output logging file
    log_conf.add_log_file(cfg["output_dir"])
    logging.info("Output directory: %s", cfg["output_dir"])

    logging.info("*** Demcompare ***")
    logging.debug("Demcompare configuration: %s", cfg)

    input_ref, input_sec = load_input_dems(cfg)

    # If coregistration step is present
    if "coregistration" in cfg:
        sec_name = input_sec.input_img
        ref_name = input_ref.input_img
        logging.info("Input Secondary DEM (SEC): %s", sec_name)
        logging.info("Input Reference DEM (REF): %s", ref_name)

        # Do coregistration and obtain initial
        # and final intermediate dems for stats computation
        (
            reproj_sec,
            reproj_ref,
            reproj_coreg_sec,
            reproj_coreg_ref,
        ) = run_coregistration(cfg["coregistration"], input_ref, input_sec)

        if "statistics" in cfg:
            # Compute stats after coregistration
            stats_dataset = compute_stats_after_coregistration(
                cfg["statistics"],
                reproj_coreg_sec,
                reproj_coreg_ref,
                reproj_sec,
                reproj_ref,
            )

            # Coreg + Stats Report
            logging.info("[Coregistration + Stats Report]")
            report.generate_report(
                working_dir=cfg["output_dir"],
                stats_dataset=stats_dataset,
                sec_name=reproj_sec.input_img,
                ref_name=reproj_ref.input_img,
                coreg_sec_name=reproj_coreg_sec.input_img,
                coreg_ref_name=reproj_coreg_ref.input_img,
                doc_dir=os.path.join(
                    cfg["output_dir"], get_out_dir("sphinx_built_doc")
                ),
                project_dir=os.path.join(
                    cfg["output_dir"], get_out_dir("sphinx_src_doc")
                ),
            )

    # If only stats is present
    elif "statistics" in cfg:
        # If both dems have been defined, compute altitude difference for stats
        if input_ref and input_sec:
            reproj_sec, reproj_ref, _ = reproject_dems(
                input_sec,
                input_ref,
                sampling_source=cfg["sampling_source"]
                if "sampling_source" in cfg
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
            if "classification_layer" in cfg:
                if "fusion" in cfg["classification_layers"]:
                    verify_fusion_layers(
                        reproj_ref, cfg["classification_layers"], support="ref"
                    )
                    verify_fusion_layers(
                        reproj_sec, cfg["classification_layers"], support="sec"
                    )
            # Define stats_dem as the altitude difference
            stats_dem = compute_alti_diff_for_stats(reproj_ref, reproj_sec)
        else:
            # Compute slope and add it as a classification_layer
            # in case a classification of type slope is required
            input_ref = compute_dem_slope(input_ref)
            # If defined, verify fusion layers according to the cfg
            if "classification_layer" in cfg:
                if "fusion" in cfg["classification_layers"]:
                    verify_fusion_layers(
                        input_ref, cfg["classification_layers"], support="ref"
                    )
            # Define stats_dem as the input dem
            stats_dem = input_ref

        # Create StatsComputation object
        stats_processing = StatsProcessing(cfg["statistics"], stats_dem)
        # Compute stats according to the input stats configuration
        stats_dataset = stats_processing.compute_stats()

        # Stats Report
        # Save stats_dem
        save_dem(
            stats_dem,
            os.path.join(
                cfg["output_dir"], get_out_file_path("dem_for_stats.tif")
            ),
        )
        logging.info("[Stats Report]")
        report.generate_report(
            working_dir=cfg["output_dir"],
            stats_dataset=stats_dataset,
            doc_dir=os.path.join(
                cfg["output_dir"], get_out_dir("sphinx_built_doc")
            ),
            project_dir=os.path.join(
                cfg["output_dir"], get_out_dir("sphinx_src_doc")
            ),
        )


def load_input_dems(cfg: Dict) -> Tuple[xr.Dataset, Union[None, xr.Dataset]]:
    """
    Loads the input dems according to the input cfg

    :param cfg: input configuration
    :type cfg: Dict
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
    cfg: Dict, input_ref: xr.Dataset, input_sec: xr.Dataset
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Runs the dems coregistration

    :param cfg: coregistration configuration
    :type cfg: Dict
    :param input_ref: input ref
    :type input_ref: xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :param input_sec: input dem
    :type input_sec: xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :return: reproj_sec, reproj_ref, reproj_coreg_sec, reproj_coreg_ref
    :rtype:   Tuple(xr.Dataset, xr.dataset, xr.Dataset, xr.dataset)
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
    coreg_sec = transformation.apply_transform(input_sec)
    # Get demcompare_results dict
    demcompare_results = coregistration_.demcompare_results

    # Save the coregistered DEM
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
    reproj_ref = coregistration_.reproj_ref
    reproj_sec = coregistration_.reproj_sec
    reproj_coreg_ref = coregistration_.reproj_coreg_ref
    reproj_coreg_sec = coregistration_.reproj_coreg_sec

    return (
        reproj_sec,
        reproj_ref,
        reproj_coreg_sec,
        reproj_coreg_ref,
    )


def compute_stats_after_coregistration(
    cfg: Dict[str, dict],
    coreg_sec: xr.Dataset,
    coreg_ref: xr.Dataset,
    initial_sec: xr.Dataset = None,
    initial_ref: xr.Dataset = None,
) -> StatsDataset:
    """
    Compute stats after coregistration

    For the initial_dh and final_dh the alti_diff plot, the
    cdf and the pdf are computed if the output_dir has
    been specified in order to evaluate the
    coregistration effect.

    For the final_dh, the different classification layers
    and metrics specified in the input cfg are also computed.

    :param cfg: configuration dictionary
    :type cfg: dict
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
    :param initial_sec: optional initial dem
                to align xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type initial_sec: xr.Dataset
    :param initial_ref: optional initial reference dem xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type initial_ref: xr.Dataset
    :return: StatsDataset
    :rtype: StatsDataset
    """
    logging.info("[Stats]")
    logging.info("# Altimetric error stats generation")
    # Initial stats --------------------------------------------

    # Compute altitude diff
    initial_altitude_diff = compute_dems_diff(initial_ref, initial_sec)

    # Obtain output paths
    (
        dem_path,
        plot_file_path,
        plot_path_cdf,
        csv_path_cdf,
        plot_path_pdf,
        csv_path_pdf,
    ) = helpers_init.get_output_files_paths(
        cfg["output_dir"], "initial_dem_diff"
    )
    # Compute and save initial altitude diff image plots
    compute_and_save_image_plots(
        initial_altitude_diff,
        plot_file_path,
        title="initial [REF - DEM] differences",
        dem_path=dem_path,
    )

    # Create StatsComputation object for the initial_dh
    # We do not need any classification_layer for the initial_dh
    initial_stats_cfg = {
        "remove_outliers": "False",
        "output_dir": cfg["output_dir"],
    }
    stats_processing_initial = StatsProcessing(
        initial_stats_cfg, initial_altitude_diff, input_diff=True
    )

    # For the initial_dh, compute cdf and pdf stats
    # on the global classification layer (diff, pdf, cdf)
    plot_metrics = [
        {
            "cdf": {
                "remove_outliers": "False",
                "output_plot_path": plot_path_cdf,
                "output_csv_path": csv_path_cdf,
            }
        },
        {
            "pdf": {
                "remove_outliers": "False",
                "output_plot_path": plot_path_pdf,
                "output_csv_path": csv_path_pdf,
            }
        },
    ]
    stats_processing_initial.compute_stats(
        classification_layer=["global"],
        metrics=plot_metrics,  # type: ignore
    )

    # Final stats --------------------------------------------

    # Compute slope and add it as a classification_layer in case
    # a classification of type slope is required
    # The ref is considered the main classification,
    # the slope of the sec dem will be used for the intersection-exclusion
    coreg_ref = compute_dem_slope(coreg_ref)
    coreg_sec = compute_dem_slope(coreg_sec)

    # If defined, verify fusion layers according to the cfg
    if "classification_layer_masks" in cfg:
        if "fusion" in cfg["classification_layers"]:
            verify_fusion_layers(
                coreg_ref, cfg["classification_layers"], support="ref"
            )
            verify_fusion_layers(
                coreg_sec, cfg["classification_layers"], support="sec"
            )

    # Compute altitude diff
    final_altitude_diff = compute_alti_diff_for_stats(coreg_ref, coreg_sec)

    # Create StatsComputation object for the final_dh
    final_stats_cfg = copy.deepcopy(cfg)
    stats_processing_final = StatsProcessing(
        final_stats_cfg, final_altitude_diff, input_diff=True
    )

    # Obtain output paths
    (
        dem_path,
        plot_file_path,
        plot_path_cdf,
        csv_path_cdf,
        plot_path_pdf,
        csv_path_pdf,
    ) = helpers_init.get_output_files_paths(cfg["output_dir"], "final_dem_diff")

    # Compute and save final altitude diff image plots
    compute_and_save_image_plots(
        final_altitude_diff,
        plot_file_path,
        title="final [REF - DEM] differences",
        dem_path=dem_path,
    )
    # For the final_dh, first compute plot stats on the
    # global classification layer (diff, pdf, cdf)
    plot_metrics = [
        {
            "cdf": {
                "remove_outliers": "False",
                "output_plot_path": plot_path_cdf,
                "output_csv_path": csv_path_cdf,
            }
        },
        {
            "pdf": {
                "remove_outliers": "False",
                "output_plot_path": plot_path_pdf,
                "output_csv_path": csv_path_pdf,
            }
        },
    ]
    stats_processing_final.compute_stats(
        classification_layer=["global"],
        metrics=plot_metrics,  # type: ignore
    )

    # For the final_dh, also compute all classif layer default metric stats
    stats_dataset = stats_processing_final.compute_stats()
    return stats_dataset
