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
from . import initialization, report, stats
from .coregistration import Coregistration
from .dem_tools import compute_dems_diff, load_dem, save_dem
from .output_tree_design import get_otd_dirs, get_out_dir, get_out_file_path

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
DEFAULT_STEPS = ["coregistration", "stats", "report"]


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


# TODO: to be affected by the refacto stats
def compute_report(
    cfg: Dict,
    demcompare_results: Dict,
    steps: List[str],
    dem_to_align_name: str,
    ref_name: str,
    coreg_dem_to_align_name: str,
    coreg_ref_name: str,
):
    """
    Create html and pdf report through sphinx generation

    :param cfg: configuration dictionary
    :type cfg: dict
    :param steps: pipeline steps
    :type steps: List
    :param dem_to_align_name: dem_to_align raster name
    :type dem_to_align_name: str
    :param ref_name: reference dem_to_align raster name
    :type ref_name: str
    :param coreg_dem_to_align_name: coreg_dem_to_align name
    :type coreg_dem_to_align_name: str
    :param coreg_ref_name: coreg_ref name
    :type coreg_ref_name: str
    :return: None
    """
    if coreg_dem_to_align_name is None:
        coreg_dem_to_align_name = ""
    if coreg_ref_name is None:
        coreg_ref_name = ""

    if "report" in steps:
        logging.info("[Report]")
        report.generate_report(
            cfg["output_dir"],
            dem_to_align_name,
            ref_name,
            coreg_dem_to_align_name,
            coreg_ref_name,
            demcompare_results["stats_results"]["partitions"],
            os.path.join(cfg["output_dir"], get_out_dir("sphinx_built_doc")),
            os.path.join(cfg["output_dir"], get_out_dir("sphinx_src_doc")),
        )


# TODO: to be affected by the refacto stats
def compute_stats(
    cfg: Dict[str, dict],
    demcompare_results: Dict[str, dict],
    dem_to_align: xr.Dataset,
    ref: xr.Dataset,
    final_dh: xr.Dataset,
    display: bool = False,
    final_json_file: str = None,
):
    """
    Compute Stats on final_dh

    :param cfg: configuration dictionary
    :type cfg: dict
    :param dem_to_align: dem to align xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type dem_to_align: xr.Dataset
    :param ref: reference dem xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type ref: xr.Dataset
    :param final_dh: initial alti diff
    :param final_dh: initial alti diff xr.Dataset containing the variables :
            - im : 2D (row, col) xarray.DataArray float32
            - trans: 1D (trans_len) xarray.DataArray

    :param final_dh: initial alti diff xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type final_dh: xr.Dataset
    :param display: choose between plot show and plot save
    :type display: boolean
    :param final_json_file: filename of final_cfg
    :type final_json_file: str
    :return:
    """
    logging.info("[Stats]")
    demcompare_results["stats_results"] = {}
    demcompare_results["stats_results"]["images"] = {}
    demcompare_results["stats_results"]["images"]["list"] = []

    logging.info("# DEM diff wave detection")
    stats.wave_detection(cfg, demcompare_results, final_dh)

    logging.info("# Altimetric error stats generation")
    stats.alti_diff_stats(
        cfg,
        demcompare_results,
        dem_to_align,
        ref,
        final_dh,
        display=display,
        remove_outliers=cfg["stats_opts"]["remove_outliers"],
    )
    # save results
    logging.info(
        "Save final results stats information file: {}".format(final_json_file)
    )
    logging.debug("Final stats results: {}".format(demcompare_results))
    initialization.save_config_file(final_json_file, demcompare_results)


# TODO : evolution with refacto __init__.py demcompare
def dem_coregistration(
    cfg: Dict,
    dem_to_align: xr.Dataset,
    ref: xr.Dataset,
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
    :param dem_to_align: dem_to_align xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type dem_to_align: xr.Dataset
    :param ref: reference dem raster xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type ref: xr.Dataset
    :param final_json_file: filename of final_json_file
    :type final_json_file: str
    :return: coreg_dem_to_align xr.Dataset, coreg_ref xr.Dataset,
                final_dh xr.Dataset, coreg_state. The xr.Datasets
                containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :rtype: xr.Dataset, xr.Dataset, xr.Dataset, bool
    """

    logging.info("[Coregistration]")
    # TODO: refactoring function, coreg_state kept for consistency
    coreg_state = True

    # Create coregistration object
    coregistration_ = Coregistration(cfg["coregistration"])
    # Compute coregistration to get applicable transformation object
    transformation = coregistration_.compute_coregistration(dem_to_align, ref)

    # Apply coregistration offsets to the original DEM and store it
    coreg_dem_to_align = transformation.apply_transform(dem_to_align)

    # If output_dir is defined, save the coregistered DEM
    if "output_dir" in cfg:
        # Save coregistered DEM
        # - coreg_DEM.tif -> coregistered dem_to_align
        save_dem(
            coreg_dem_to_align,
            os.path.join(cfg["output_dir"], get_out_file_path("coreg_DEM.tif")),
        )

    # TODO: do this part only if stats are to be computed

    # Get demcompare_results dict
    demcompare_results = coregistration_.demcompare_results

    # Get internal dems
    reproj_ref = coregistration_.reproj_ref
    reproj_dem_to_align = coregistration_.reproj_dem_to_align
    reproj_coreg_ref = coregistration_.reproj_coreg_ref
    reproj_coreg_dem_to_align = coregistration_.reproj_coreg_dem_to_align
    # Compute initial_dh
    initial_dh = compute_dems_diff(reproj_ref, reproj_dem_to_align)
    # Compute final_dh
    final_dh = compute_dems_diff(reproj_coreg_ref, reproj_coreg_dem_to_align)

    # TODO: do this part only if stats are to be computed

    if "output_dir" in cfg:
        # Saves initial altitude difference to file system
        initial_dh = save_dem(
            initial_dh,
            os.path.join(
                cfg["output_dir"], get_out_file_path("initial_dh.tif")
            ),
        )
        # Saves final altitude difference to file system
        final_dh = save_dem(
            final_dh,
            os.path.join(cfg["output_dir"], get_out_file_path("final_dh.tif")),
        )

    # TODO: to be modified by the refacto stats
    stats.dem_diff_plot(
        initial_dh,
        title="Initial [REF - DEM] differences",
        plot_file=os.path.join(
            cfg["output_dir"], get_out_file_path("initial_dem_diff.png")
        ),
        display=False,
    )
    stats.dem_diff_cdf_plot(
        initial_dh,
        title="Initial [REF - DEM] differences CDF",
        plot_file=os.path.join(
            cfg["output_dir"], get_out_file_path("initial_dem_diff_cdf.png")
        ),
        display=False,
    )
    stats.dem_diff_pdf_plot(
        initial_dh,
        title="Elevation difference Histogram",
        plot_file=os.path.join(
            cfg["output_dir"], get_out_file_path("initial_dem_diff_pdf.png")
        ),
        display=False,
    )
    # TODO: may be modified by the refacto script demcompare
    # saves results here in case next step fails
    initialization.save_config_file(final_json_file, demcompare_results)

    # TODO: the returns will be modified by the refacto stats
    return (
        reproj_coreg_dem_to_align,
        reproj_coreg_ref,
        final_dh,
        coreg_state,
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
        # Save output_dir parameter in "coregistration" and/or "stats_opts" dict
        if "coregistration" in cfg:
            cfg["coregistration"]["output_dir"] = output_dir
        if "stats_opts" in cfg:
            cfg["stats_opts"]["output_dir"] = output_dir
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
            if "stats_opts" in cfg:
                cfg["stats_opts"]["sampling_source"] = cfg["coregistration"][
                    "sampling_source"
                ]

    # TODO: to be modified by stats refactoring
    initialization.initialization_stats_opts(cfg)
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
    )

    dem_to_align_orig = load_dem(
        cfg["input_dem_to_align"]["path"],
        no_data=(
            cfg["input_dem_to_align"]["nodata"]
            if "nodata" in cfg["input_dem_to_align"]
            else None
        ),
        geoid_georef=(
            cfg["input_dem_to_align"]["geoid_georef"]
            if "geoid_georef" in cfg["input_dem_to_align"]
            else False
        ),
        geoid_path=(
            cfg["input_dem_to_align"]["geoid_path"]
            if "geoid_path" in cfg["input_dem_to_align"]
            else None
        ),
        zunit=(
            cfg["input_dem_to_align"]["zunit"]
            if "zunit" in cfg["input_dem_to_align"]
            else "m"
        ),
        input_roi=(
            cfg["input_dem_to_align"]["roi"]
            if "roi" in cfg["input_dem_to_align"]
            else True
        ),
    )

    dem_to_align_name = dem_to_align_orig.input_img
    ref_name = ref_orig.input_img
    logging.info("Input Tested DEM (DEM): {}".format(dem_to_align_name))
    logging.info("Input Reference DEM (REF): {}".format(ref_name))

    #
    # Coregister both DSMs together and compute final differences
    #
    (
        interm_coreg_dem_to_align,
        interm_coreg_ref,
        final_dh,
        coreg_state,
        demcompare_results,
    ) = dem_coregistration(
        cfg,
        dem_to_align_orig,
        ref_orig,
        final_json_file=final_json_file,
    )
    if coreg_state:
        #
        #  Plot/Save final dh img and cdf if exists
        #
        logging.info(
            "Input Coreg Tested DEM (COREG_DEM): {}".format(
                interm_coreg_dem_to_align.input_img
            )
        )
        logging.info(
            "Input Coreg Reference DEM (COREG_REF): {}".format(
                interm_coreg_ref.input_img
            )
        )
        logging.info(
            "Final diff DEM (COREG_REF - COREG_DEM): {}".format(
                final_dh.input_img
            )
        )
        stats.dem_diff_plot(
            final_dh,
            title="Final [COREG_REF - COREG_DEM] differences",
            plot_file=os.path.join(
                cfg["output_dir"], get_out_file_path("final_dem_diff.png")
            ),
            display=display,
        )
        stats.dem_diff_cdf_plot(
            final_dh,
            title="Final [COREG_REF - COREG_DEM] differences CDF",
            plot_file=os.path.join(
                cfg["output_dir"], get_out_file_path("final_dem_diff_cdf.png")
            ),
            display=display,
        )
        stats.dem_diff_pdf_plot(
            final_dh,
            title="Elevation difference Histogram",
            plot_file=os.path.join(
                cfg["output_dir"], get_out_file_path("final_dem_diff_pdf.png")
            ),
            display=display,
        )

    #
    # Compute stats
    #
    compute_stats(
        cfg,
        demcompare_results,
        interm_coreg_dem_to_align,
        interm_coreg_ref,
        final_dh,
        display=display,
        final_json_file=final_json_file,
    )

    #
    # Compute reports
    #
    compute_report(
        cfg,
        demcompare_results,
        steps,
        dem_to_align_name,
        ref_name,
        interm_coreg_dem_to_align.input_img,
        interm_coreg_ref.input_img,
    )


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
