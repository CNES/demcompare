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

Note : this version of the __init__.py is transitional to keep the tests valid
during the refactoring, the structure and aim will evolve after dem_tools,
coreg, stats refactoring.
"""

# Standard imports
import json
import logging
import logging.config
import os
import sys
import traceback
from typing import Dict, List

# Third party imports
import matplotlib as mpl
import xarray as xr

# Demcompare imports
from . import initialization, report
from .coregistration import Coregistration
from .dem_tools import (
    compute_alti_diff_for_stats,
    compute_and_save_image_plots,
    compute_dem_slope,
    compute_dems_diff,
    compute_waveform,
    load_dem,
    save_dem,
    verify_fusion_layers,
)
from .output_tree_design import get_otd_dirs, get_out_dir, get_out_file_path
from .stats_processing import StatsProcessing

# ** VERSION **
# pylint: disable=import-error,no-name-in-module
# Depending on python version get importlib standard lib or backported package
# TODO: remove pythons < 3.8
if sys.version_info[:2] >= (3, 8):
    # when python3 > 3.8
    from importlib.metadata import PackageNotFoundError  # pragma: no cover
    from importlib.metadata import version
else:
    from importlib_metadata import PackageNotFoundError  # pragma: no cover
    from importlib_metadata import version
# Get demcompare package version (installed from setuptools_scm)
try:
    __version__ = version("demcompare")
except PackageNotFoundError:
    __version__ = "unknown"  # pragma: no cover
finally:
    del version, PackageNotFoundError

# ** STEPS **
DEFAULT_STEPS = ["coregistration", "statistics", "report"]


def setup_logging(
    logconf_path="logging.json",
    default_level=logging.INFO,
):
    """
    Setup the logging configuration
    If logconf_path is found, set the json logging configuration
    Else put default_level

    :param logconf_path: path to the configuration file
    :type logconf_path: string
    :param default_level: default level
    :type default_level: logging level
    """
    logconf_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), logconf_path)
    )
    if os.path.exists(logconf_path):
        with open(logconf_path, "rt", encoding="utf8") as logconf_file:
            config = json.load(logconf_file)
        # Set config and force default_level from command_line
        logging.config.dictConfig(config)
        logging.getLogger().setLevel(default_level)
    else:
        # take default python config with default_level from command line
        logging.basicConfig(level=default_level)


# TODO: to be modified by the refacto script demcompare
def compute_report(
    cfg: Dict,
    steps: List[str],
    stats_dataset,
    sec_name: str,
    ref_name: str,
    coreg_sec_name: str,
    coreg_ref_name: str,
):
    """
    Create html and pdf report through sphinx generation

    :param cfg: configuration dictionary
    :type cfg: dict
    :param steps: pipeline steps
    :type steps: List
    :param sec_name: sec raster name
    :type sec_name: str
    :param ref_name: reference sec raster name
    :type ref_name: str
    :param coreg_sec_name: sec name
    :type coreg_sec_name: str
    :param coreg_ref_name: ref name
    :type coreg_ref_name: str
    :return: None
    """
    if coreg_sec_name is None:
        coreg_sec_name = ""
    if coreg_ref_name is None:
        coreg_ref_name = ""

    if "report" in steps:
        logging.info("[Report]")
        report.generate_report(
            working_dir=cfg["output_dir"],
            stats_dataset=stats_dataset,
            sec_name=sec_name,
            ref_name=ref_name,
            coreg_sec_name=coreg_sec_name,
            coreg_ref_name=coreg_ref_name,
            doc_dir=os.path.join(
                cfg["output_dir"], get_out_dir("sphinx_built_doc")
            ),
            project_dir=os.path.join(
                cfg["output_dir"], get_out_dir("sphinx_src_doc")
            ),
        )


# TODO: to be modified by the refacto script demcompare
def compute_stats(
    cfg: Dict[str, dict],
    coreg_sec: xr.Dataset,
    coreg_ref: xr.Dataset,
    initial_sec: xr.Dataset = None,
    initial_ref: xr.Dataset = None,
):
    """
    Compute Stats

    :param cfg: configuration dictionary
    :type cfg: dict
    :param coreg_sec: coreg dem to align xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type coreg_sec: xr.Dataset
    :param coreg_ref: coreg reference dem xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :param initial_sec: optional initial dem
                to align xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type initial_sec: xr.Dataset
    :param initial_ref: optional initial reference dem xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type initial_ref: xr.Dataset
    :return:
    """
    logging.info("[Stats]")

    logging.info("# DEM diff wave detection")

    logging.info("# Altimetric error stats generation")

    # Cfg for the initial stats --------------------------------------------

    # Compute altitude diff
    initial_altitude_diff = compute_dems_diff(initial_ref, initial_sec)
    # Compute and save waveform tifs
    compute_waveform(initial_altitude_diff, cfg["output_dir"])

    # Obtain output paths
    (
        dem_path,
        plot_file_path,
        plot_path_cdf,
        csv_path_cdf,
        plot_path_pdf,
        csv_path_pdf,
    ) = initialization.compute_output_files_paths(
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
    # We do not need any classification_layers for the initial_dh
    initial_stats_cfg = {
        "remove_outliers": "False",
        "output_dir": cfg["output_dir"],
    }
    stats_processing_initial = StatsProcessing(
        initial_stats_cfg, initial_altitude_diff
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
        classification_layer=["global"], metrics=plot_metrics
    )

    # Cfg for the final stats --------------------------------------------

    # Compute slope and add it as a classification_layer
    # The ref name as it is considered the main classification,
    # the slope of the sec dem will be used for the intersection-exclusion
    coreg_ref = compute_dem_slope(coreg_ref)
    coreg_sec = compute_dem_slope(coreg_sec)

    # Verify fusion layers according to the cfg
    if "fusion" in cfg["statistics"]["classification_layers"]:
        coreg_ref = verify_fusion_layers(
            coreg_ref, cfg["statistics"]["classification_layers"], name="ref"
        )
        coreg_sec = verify_fusion_layers(
            coreg_sec, cfg["statistics"]["classification_layers"], name="sec"
        )

    # Compute altitude diff
    final_altitude_diff = compute_alti_diff_for_stats(coreg_ref, coreg_sec)
    # Compute and save waveform tifs
    compute_waveform(final_altitude_diff, cfg["output_dir"])

    # Obtain output paths
    (
        dem_path,
        plot_file_path,
        plot_path_cdf,
        csv_path_cdf,
        plot_path_pdf,
        csv_path_pdf,
    ) = initialization.compute_output_files_paths(
        cfg["output_dir"], "final_dem_diff"
    )

    # Compute and save initial altitude diff image plots
    compute_and_save_image_plots(
        final_altitude_diff,
        plot_file_path,
        title="final [REF - DEM] differences",
        dem_path=dem_path,
    )

    # Create StatsComputation object for the final_dh
    final_stats_cfg = cfg["statistics"]
    stats_processing_final = StatsProcessing(
        final_stats_cfg, final_altitude_diff
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
        classification_layer=["global"], metrics=plot_metrics
    )

    # For the final_dh, also compute all classif layer default metric stats
    stats_dataset = stats_processing_final.compute_stats()
    return stats_dataset


# TODO: to be modified by the refacto script demcompare
def dem_coregistration(
    cfg: Dict,
    sec: xr.Dataset,
    ref: xr.Dataset,
    cfg_stats: bool = False,
    final_json_file: str = None,
):
    """
    Coregister two DEMs together and compute alti differences
    (before and after coregistration).

    This can be view as a two step process:
    - plani rectification computation
    - alti differences computation

    :param cfg: configuration dictionary
    :type cfg: dict
    :param sec: sec xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type sec: xr.Dataset
    :param ref: reference dem raster xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type ref: xr.Dataset
    :param cfg_stats: is stats option is in the input cfg
    :type cfg_stats: bool
    :param final_json_file: filename of final_json_file
    :type final_json_file: str
    :return: sec xr.Dataset, ref xr.Dataset,
                final_dh xr.Dataset, coreg_state. The xr.Datasets
                containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :rtype: xr.Dataset, xr.Dataset, xr.Dataset, bool
    """

    logging.info("[Coregistration]")

    # Create coregistration object
    coregistration_ = Coregistration(cfg["coregistration"])
    # Compute coregistration to get applicable transformation object
    transformation = coregistration_.compute_coregistration(sec, ref)

    # Apply coregistration offsets to the original DEM and store it
    coreg_sec = transformation.apply_transform(sec)

    # If output_dir is defined, save the coregistered DEM
    if "output_dir" in cfg:
        # Save coregistered DEM
        # - coreg_DEM.tif -> coregistered sec
        save_dem(
            coreg_sec,
            os.path.join(cfg["output_dir"], get_out_file_path("coreg_DEM.tif")),
        )

    # Get demcompare_results dict
    demcompare_results = coregistration_.demcompare_results

    reproj_ref = None
    reproj_sec = None
    reproj_coreg_ref = None
    reproj_coreg_sec = None
    if cfg_stats:
        # Get internal dems
        reproj_ref = coregistration_.reproj_ref
        reproj_sec = coregistration_.reproj_sec
        reproj_coreg_ref = coregistration_.reproj_coreg_ref
        reproj_coreg_sec = coregistration_.reproj_coreg_sec

    # TODO: may be modified by the refacto script demcompare
    # saves results here in case next step fails
    initialization.save_config_file(final_json_file, demcompare_results)

    return (
        reproj_sec,
        reproj_ref,
        reproj_coreg_sec,
        reproj_coreg_ref,
        demcompare_results,
    )


# TODO: must be modified by the refacto script demcompare
def compute_initialization(config_json: str) -> Dict:
    """
    Compute demcompare initialization process :
    Configuration copy, checking, create output dir tree
    and initial output content.

    :param config_json: Config json file name
    :type config_json: str
    :return: cfg
    :rtype: Dict[str, Dict]
    """

    # Read the json configuration file
    # (and update inputs path with absolute path)
    cfg = initialization.read_config_file(config_json)

    # Checks input config
    initialization.check_input_parameters(cfg)

    # Create output directory and update config
    if "output_dir" in cfg:
        output_dir = os.path.abspath(cfg["output_dir"])
        cfg["output_dir"] = output_dir
        # Save output_dir parameter in "coregistration" and/or "statistics" dict
        if "coregistration" in cfg:
            cfg["coregistration"]["output_dir"] = output_dir
        if "statistics" in cfg:
            cfg["statistics"]["output_dir"] = output_dir

        # Create output_dir
        initialization.mkdir_p(cfg["output_dir"])

        # Save initial config with inputs absolute paths into output_dir
        initialization.save_config_file(
            os.path.join(cfg["output_dir"], os.path.basename(config_json)), cfg
        )

        # create output tree dirs for each directory
        for directory in get_otd_dirs(cfg["otd"]):
            initialization.mkdir_p(os.path.join(cfg["output_dir"], directory))

    # Force the sampling_source of the coregistration step into the stats step
    if "coregistration" in cfg:
        if "sampling_source" in cfg["coregistration"]:
            if "statistics" in cfg:
                cfg["statistics"]["sampling_source"] = cfg["coregistration"][
                    "sampling_source"
                ]
    return cfg


# TODO: to be modified by the refacto script demcompare
def run_tile(json_file: str, steps: List[str] = None, display=False):
    """
    Demcompare execution for a single tile

    :param json_file: Input Json configuration file (mandatory)
    :type json_file: str
    :param steps: Steps to execute (default: all)
    :type steps: List[str]
    :param display: Choose Plot show or plot save (default).
    :type display: bool
    """

    # Set steps to default if None
    if steps is None:
        steps = DEFAULT_STEPS
    cfg_stats = "statistics" in steps
    #
    # Initialization
    #
    cfg = compute_initialization(json_file)

    logging.info("*** Demcompare ***")
    logging.info("Output directory: {}".format(cfg["output_dir"]))
    logging.debug("Demcompare configuration: {}".format(cfg))

    sys.stdout.flush()
    if display is False:
        # if display is False we have to tell matplotlib to cancel it
        mpl.use("Agg")

    # Set final_json_file name
    final_json_file = os.path.join(
        cfg["output_dir"], get_out_file_path("demcompare_results.json")
    )
    #
    # Create datasets
    #
    ref_orig = load_dem(
        cfg["input_ref"]["path"],
        no_data=(
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
            cfg["input_ref"]["roi"] if "roi" in cfg["input_ref"] else True
        ),
        classification_layers=(
            cfg["input_ref"]["classification_layers"]
            if "classification_layers" in cfg["input_ref"]
            else None
        ),
    )

    sec_orig = load_dem(
        cfg["input_sec"]["path"],
        no_data=(
            cfg["input_sec"]["nodata"] if "nodata" in cfg["input_sec"] else None
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
            cfg["input_sec"]["zunit"] if "zunit" in cfg["input_sec"] else "m"
        ),
        input_roi=(
            cfg["input_sec"]["roi"] if "roi" in cfg["input_sec"] else True
        ),
        classification_layers=(
            cfg["input_sec"]["classification_layers"]
            if "classification_layers" in cfg["input_sec"]
            else None
        ),
    )

    sec_name = sec_orig.input_img
    ref_name = ref_orig.input_img
    logging.info("Input Tested secondary DEM (SEC): {}".format(sec_name))
    logging.info("Input Reference reference DEM (REF): {}".format(ref_name))

    #
    # Coregister both DSMs together and compute final differences
    #
    (
        interm_sec,
        interm_ref,
        interm_coreg_sec,
        interm_coreg_ref,
        demcompare_results,
    ) = dem_coregistration(
        cfg,
        sec_orig,
        ref_orig,
        cfg_stats,
        final_json_file=final_json_file,
    )

    #
    # Compute stats
    #
    if cfg_stats:

        stats_dataset = compute_stats(
            cfg,
            interm_coreg_sec,
            interm_coreg_ref,
            interm_sec,
            interm_ref,
        )
        #
        # Compute reports
        #
        compute_report(
            cfg,
            steps,
            stats_dataset,
            sec_name,
            ref_name,
            interm_coreg_sec.input_img,
            interm_coreg_ref.input_img,
        )

    logging.info(
        "Save final results stats information file: {}".format(final_json_file)
    )
    logging.debug("Final stats results: {}".format(demcompare_results))
    initialization.save_config_file(final_json_file, demcompare_results)


# TODO: to be modified by the refacto script demcompare
def run(
    json_file: str,
    steps: List[str] = None,
    display: bool = False,
    loglevel=logging.WARNING,
):
    """
    Demcompare main execution for all tiles.
    Call run_tile() function for each tile.

    :param json_file: Input Json configuration file (mandatory)
    :type json_file: str
    :param steps: Steps to execute (default: all)
    :type steps: List[str]
    :param display: Choose Plot show or plot save (default).
    :type display: bool
    :param loglevel: Choose Loglevel (default: WARNING)
    :type loglevel: logging.WARNING
    """

    # Set steps to default if None
    if steps is None:
        steps = DEFAULT_STEPS

    #
    # Initialization
    #
    setup_logging(default_level=loglevel)
    cfg = compute_initialization(json_file)
    if display is False:
        # if display is False we have to tell matplotlib to cancel it
        mpl.use("Agg")

    #
    # Get back tiles
    #
    if "tile_size" not in cfg:
        tiles = [{"json": json_file}]
    else:
        tiles = initialization.divide_images(cfg)

    #
    # Run classic steps by tiles
    # (there can be just one tile which could be the whole image)
    #
    for tile in tiles:
        try:
            run_tile(
                tile["json"],
                steps,
                display=display,
            )
        except Exception as error:  # pylint: disable=broad-except
            traceback.print_exc()
            logging.error(
                "Error encountered for tile: {} -> {}".format(tile, error)
            )
