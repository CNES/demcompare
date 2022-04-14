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
DEMcompare init module file.
DEMcompare aims at coregistering and comparing two Digital Elevation Models(DEM)
"""

# Standard imports
import copy
import json
import logging
import logging.config
import os
import sys
import traceback
from typing import Dict, List

# Third party imports
import matplotlib as mpl
import numpy as np
import xarray as xr

# DEMcompare imports
from . import coregistration, initialization, report, stats
from .dem_loading_tools import (
    create_dataset_from_dataset,
    load_dem,
    reproject_dems,
    save_dataset_to_tif,
)
from .output_tree_design import get_otd_dirs, get_out_dir, get_out_file_path

# ** VERSION **
# pylint: disable=import-error,no-name-in-module
# Depending on python version get importlib standard lib or backported package
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
    logconf_path="demcompare/logging.json",
    default_level=logging.WARNING,
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
    if os.path.exists(logconf_path):
        with open(logconf_path, "rt", encoding="utf8") as logconf_file:
            config = json.load(logconf_file)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def compute_report(
    cfg: Dict,
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
    if "report" in steps:
        print("\n[Report]")
        report.generate_report(
            cfg["outputDir"],
            dem_to_align_name,
            ref_name,
            coreg_dem_to_align_name,
            coreg_ref_name,
            cfg["stats_results"]["partitions"],
            os.path.join(cfg["outputDir"], get_out_dir("sphinx_built_doc")),
            os.path.join(cfg["outputDir"], get_out_dir("sphinx_src_doc")),
        )


def compute_stats(
    cfg: Dict[str, dict],
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

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type dem_to_align: xr.Dataset
    :param ref: reference dem xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type ref: xr.Dataset
    :param final_dh: initial alti diff xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type final_dh: xr.Dataset
    :param display: choose between plot show and plot save
    :type display: boolean
    :param final_json_file: filename of final_cfg
    :type final_json_file: str
    :return:
    """
    print("\n[Stats]")
    cfg["stats_results"] = {}
    cfg["stats_results"]["images"] = {}
    cfg["stats_results"]["images"]["list"] = []

    print("# DEM diff wave detection")
    stats.wave_detection(cfg, final_dh)

    print("# Altimetric error stats generation")
    stats.alti_diff_stats(
        cfg,
        dem_to_align,
        ref,
        final_dh,
        display=display,
        remove_outliers=cfg["stats_opts"]["remove_outliers"],
    )
    # save results
    print("Save final results stats information file:")
    print(final_json_file)
    initialization.save_config_file(final_json_file, cfg)


def compute_coregistration(
    cfg: Dict,
    steps: List[str],
    dem_to_align: xr.Dataset,
    ref: xr.Dataset,
    initial_dh: xr.Dataset,
    final_cfg: Dict[str, dict] = None,
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
    :param steps: pipeline steps
    :type steps: List
    :param dem_to_align: dem_to_align xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type dem_to_align: xr.Dataset
    :param ref: reference dem raster xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type ref: xr.Dataset
    :param initial_dh: initial difference xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type initial_dh: xr.Dataset
    :param final_cfg: filename of final_cfg
    :type final_cfg: dict
    :param final_json_file: filename of final_json_file
    :type final_json_file: str
    :return: coreg_dem_to_align xr.Dataset, coreg_ref xr.Dataset,
                final_dh xr.Dataset, coreg_state. The xr.Datasets
                containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :rtype: xr.Dataset, xr.Dataset, xr.Dataset, bool
    """
    print("[Coregistration]")
    coreg_state = False
    if "coregistration" in steps:
        print("# Nuth & Kaab coregistration")
        coreg_state = True

        (
            coreg_dem_to_align,
            coreg_ref,
            final_dh,
        ) = coregistration.coregister_and_compute_alti_diff(
            cfg, dem_to_align, ref
        )

        # saves results here in case next step fails
        initialization.save_config_file(final_json_file, cfg)

    else:
        # If cfg from a previous run, get previous conf
        print("No coregistration step stated.")
        if (
            final_cfg is not None
            and "plani_results" in final_cfg
            and "alti_results" in final_cfg
        ):
            print("Previous coregistration found: get configuration")
            coreg_state = True

            cfg["plani_results"] = final_cfg["plani_results"]
            cfg["alti_results"] = final_cfg["alti_results"]
            coreg_dem_to_align = load_dem(
                str(cfg["alti_results"]["rectifiedDSM"]["path"]),
                nodata=(
                    cfg["alti_results"]["rectifiedDSM"]["nodata"]
                    if "nodata" in cfg["alti_results"]["rectifiedDSM"]
                    else None
                ),
            )
            coreg_ref = load_dem(
                str(cfg["alti_results"]["rectifiedRef"]["path"]),
                nodata=(
                    cfg["alti_results"]["rectifiedRef"]["nodata"]
                    if "nodata" in cfg["alti_results"]["rectifiedRef"]
                    else None
                ),
            )
            final_dh = load_dem(
                str(cfg["alti_results"]["dzMap"]["path"]),
                nodata=cfg["alti_results"]["dzMap"]["nodata"],
            )

        else:
            # Set a default config for following steps from initial DEMs
            # No coregistration done.
            print("Set coregistration DEMs equal to input DEMs")
            coreg_state = False
            coreg_ref = ref
            coreg_dem_to_align = dem_to_align
            final_dh = initial_dh
            cfg["plani_results"] = {}
            cfg["plani_results"]["dx"] = {"bias_value": 0, "unit": "m"}
            cfg["plani_results"]["dy"] = {"bias_value": 0, "unit": "m"}
            cfg["alti_results"] = {}
            cfg["alti_results"]["rectifiedDSM"] = copy.deepcopy(cfg["inputDSM"])
            cfg["alti_results"]["rectifiedRef"] = copy.deepcopy(cfg["inputRef"])

            coreg_dem_to_align = save_dataset_to_tif(
                coreg_dem_to_align,
                os.path.join(
                    cfg["outputDir"], get_out_file_path("coreg_DEM.tif")
                ),
            )
            coreg_ref = save_dataset_to_tif(
                coreg_ref,
                os.path.join(
                    cfg["outputDir"], get_out_file_path("coreg_REF.tif")
                ),
            )
            final_dh = save_dataset_to_tif(
                final_dh,
                os.path.join(
                    cfg["outputDir"], get_out_file_path("final_dh.tif")
                ),
            )
            cfg["alti_results"]["rectifiedDSM"][
                "path"
            ] = coreg_dem_to_align.attrs["input_img"]
            cfg["alti_results"]["rectifiedRef"]["path"] = coreg_ref.attrs[
                "input_img"
            ]
            cfg["alti_results"]["rectifiedDSM"][
                "nb_points"
            ] = coreg_dem_to_align["im"].data.size
            cfg["alti_results"]["rectifiedRef"]["nb_points"] = coreg_ref[
                "im"
            ].data.size
            cfg["alti_results"]["rectifiedDSM"][
                "nb_valid_points"
            ] = np.count_nonzero(~np.isnan(coreg_dem_to_align["im"].data))
            cfg["alti_results"]["rectifiedRef"][
                "nb_valid_points"
            ] = np.count_nonzero(~np.isnan(coreg_ref["im"].data))
            cfg["alti_results"]["dzMap"] = {
                "path": final_dh.attrs["input_img"],
                "zunit": coreg_ref.attrs["zunit"].name,
                "nodata": final_dh.attrs["no_data"],
                "nb_points": final_dh["im"].data.size,
                "nb_valid_points": np.count_nonzero(
                    ~np.isnan(final_dh["im"].data.size)
                ),
            }

    return coreg_dem_to_align, coreg_ref, final_dh, coreg_state


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

    # read the json configuration file (and update inputs with absolute path)
    cfg = initialization.read_config_file(config_json)

    # create output directory and update config
    cfg["outputDir"] = os.path.abspath(cfg["outputDir"])
    initialization.mkdir_p(cfg["outputDir"])

    # Save initial config with inputs absolute paths into outputDir
    initialization.save_config_file(
        os.path.join(cfg["outputDir"], os.path.basename(config_json)), cfg
    )

    # checks config
    initialization.check_parameters(cfg)

    # create output tree dirs for each directory
    for directory in get_otd_dirs(cfg["otd"]):
        initialization.mkdir_p(os.path.join(cfg["outputDir"], directory))

    initialization.initialization_plani_opts(cfg)
    initialization.initialization_alti_opts(cfg)
    initialization.initialization_stats_opts(cfg)

    return cfg


def run_tile(json_file: str, steps: List[str] = None, display=False):
    """
    DEMcompare execution for a single tile

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

    print("*** DEMcompare ***")
    print("Working directory: {}".format(cfg["outputDir"]))
    logging.debug("Demcompare configuration: {}".format(cfg))

    sys.stdout.flush()
    if display is False:
        # if display is False we have to tell matplotlib to cancel it
        mpl.use("Agg")

    # Set final_json_file name
    final_json_file = os.path.join(
        cfg["outputDir"], get_out_file_path("final_config.json")
    )
    # Try to read json_file if exists and if a previous run was launched
    final_cfg = None
    if os.path.isfile(final_json_file):
        with open(final_json_file, "r", encoding="utf8") as file:
            final_cfg = json.load(file)

    #
    # Create datasets
    #
    ref_orig = load_dem(
        cfg["inputRef"]["path"],
        nodata=(
            cfg["inputRef"]["nodata"] if "nodata" in cfg["inputRef"] else None
        ),
        geoid_georef=(
            cfg["inputRef"]["geoid_georef"]
            if "geoid_georef" in cfg["inputRef"]
            else False
        ),
        geoid_path=(
            cfg["inputRef"]["geoid_path"]
            if "geoid_path" in cfg["inputRef"]
            else None
        ),
        zunit=(cfg["inputRef"]["zunit"] if "zunit" in cfg["inputRef"] else "m"),
        input_roi=(
            cfg["inputRef"]["roi"] if "roi" in cfg["inputRef"] else True
        ),
    )

    dem_to_align_orig = load_dem(
        cfg["inputDSM"]["path"],
        nodata=(
            cfg["inputDSM"]["nodata"] if "nodata" in cfg["inputDSM"] else None
        ),
        geoid_georef=(
            cfg["inputDSM"]["geoid_georef"]
            if "geoid_georef" in cfg["inputDSM"]
            else False
        ),
        geoid_path=(
            cfg["inputDSM"]["geoid_path"]
            if "geoid_path" in cfg["inputDSM"]
            else None
        ),
        zunit=(cfg["inputDSM"]["zunit"] if "zunit" in cfg["inputDSM"] else "m"),
        input_roi=(
            cfg["inputDSM"]["roi"] if "roi" in cfg["inputDSM"] else True
        ),
    )

    dem_to_align, ref = reproject_dems(dem_to_align_orig, ref_orig)
    print("\n# Input Elevation Models:")
    print("Tested DEM (DEM TO ALIGN): {}".format(dem_to_align.input_img))
    print("Reference DEM (REF): {}".format(ref.input_img))

    #
    # Compute initial dh, save it
    #
    initial_dh = create_dataset_from_dataset(
        ref["im"].data - dem_to_align["im"].data,
        from_dataset=dem_to_align,
        no_data=-32768,
    )
    initial_dh = save_dataset_to_tif(
        initial_dh,
        os.path.join(cfg["outputDir"], get_out_file_path("initial_dh.tif")),
    )
    print(
        "-> Initial diff DEM (REF - DEM TO ALIGN): {}\n".format(
            initial_dh.input_img
        )
    )

    #
    #  Plot/Save initial dh img and cdf
    #
    stats.dem_diff_plot(
        initial_dh,
        title="Initial [REF - DEM] differences",
        plot_file=os.path.join(
            cfg["outputDir"], get_out_file_path("initial_dem_diff.png")
        ),
        display=display,
    )
    stats.dem_diff_cdf_plot(
        initial_dh,
        title="Initial [REF - DEM] differences CDF",
        plot_file=os.path.join(
            cfg["outputDir"], get_out_file_path("initial_dem_diff_cdf.png")
        ),
        display=display,
    )
    stats.dem_diff_pdf_plot(
        initial_dh,
        title="Elevation difference Histogram",
        plot_file=os.path.join(
            cfg["outputDir"], get_out_file_path("initial_dem_diff_pdf.png")
        ),
        display=display,
    )
    #
    # Coregister both DSMs together and compute final differences
    #
    (
        coreg_dem_to_align,
        coreg_ref,
        final_dh,
        coreg_state,
    ) = compute_coregistration(
        cfg,
        steps,
        dem_to_align,
        ref,
        initial_dh,
        final_cfg=final_cfg,
        final_json_file=final_json_file,
    )
    if coreg_state:
        #
        #  Plot/Save final dh img and cdf if exists
        #
        print("# Coregistered Elevation Models:")
        print(
            "Coreg Tested DEM (COREG_DEM): {}".format(
                coreg_dem_to_align.input_img
            )
        )
        print("Coreg Reference DEM (COREG_REF): {}".format(coreg_ref.input_img))
        print(
            "--> Final diff DEM (COREG_REF - COREG_DEM): {}".format(
                final_dh.input_img
            )
        )
        stats.dem_diff_plot(
            final_dh,
            title="Final [COREG_REF - COREG_DEM] differences",
            plot_file=os.path.join(
                cfg["outputDir"], get_out_file_path("final_dem_diff.png")
            ),
            display=display,
        )
        stats.dem_diff_cdf_plot(
            final_dh,
            title="Final [COREG_REF - COREG_DEM] differences CDF",
            plot_file=os.path.join(
                cfg["outputDir"], get_out_file_path("final_dem_diff_cdf.png")
            ),
            display=display,
        )
        stats.dem_diff_pdf_plot(
            final_dh,
            title="Elevation difference Histogram",
            plot_file=os.path.join(
                cfg["outputDir"], get_out_file_path("final_dem_diff_pdf.png")
            ),
            display=display,
        )

    #
    # Compute stats
    #
    compute_stats(
        cfg,
        coreg_dem_to_align,
        coreg_ref,
        final_dh,
        display=display,
        final_json_file=final_json_file,
    )

    #
    # Compute reports
    #
    compute_report(
        cfg,
        steps,
        dem_to_align_orig.input_img,
        ref_orig.input_img,
        coreg_dem_to_align.input_img,
        coreg_ref.input_img,
    )


def run(
    json_file: str,
    steps: List[str] = None,
    display: bool = False,
    loglevel=logging.WARNING,
):
    """
    DEMcompare main execution for all tiles.
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
        except Exception as error:
            traceback.print_exc()
            print("Error encountered for tile: {} -> {}".format(tile, error))
