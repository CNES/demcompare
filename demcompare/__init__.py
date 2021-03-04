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
import shutil
import sys
import traceback

# Third party imports
import matplotlib as mpl
import numpy as np

# DEMcompare imports
from . import coregistration, initialization, report, stats
from .img_tools import load_dems, read_img, read_img_from_array, save_tif
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

    :param lo: path to the configuration file
    :type logconf_path: string
    :param default_level: default level
    :type default_level: logging level
    """
    if os.path.exists(logconf_path):
        with open(logconf_path, "rt") as logconf_file:
            config = json.load(logconf_file)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def compute_report(cfg, steps, dem, ref):
    """
    Create html and pdf report

    :param cfg: configuration dictionary
    :param dem: dem raster
    :param ref: reference dem raster to be coregistered to dem raster
    :return:
    """
    if "report" in steps:
        report.generate_report(
            cfg["outputDir"],
            dem.ds_file,
            ref.ds_file,
            cfg["stats_results"]["partitions"],
            os.path.join(cfg["outputDir"], get_out_dir("sphinx_built_doc")),
            os.path.join(cfg["outputDir"], get_out_dir("sphinx_src_doc")),
        )


def compute_stats(cfg, dem, ref, final_dh, display=False, final_json_file=None):
    """
    Compute Stats on final_dh

    :param cfg: configuration dictionary
    :param dem: dem raster
    :param ref: reference dem raster to be coregistered to dem raster
    :param final_dh: xarray.Dataset, initial alti diff
    :param display: boolean, choose between plot show and plot save
    :param final_json_file: filename of final_cfg
    :return:
    """

    cfg["stats_results"] = {}
    cfg["stats_results"]["images"] = {}
    cfg["stats_results"]["images"]["list"] = []

    stats.wave_detection(cfg, final_dh)

    stats.alti_diff_stats(
        cfg,
        dem,
        ref,
        final_dh,
        display=display,
        remove_outliers=cfg["stats_opts"]["remove_outliers"],
    )
    # save results
    with open(final_json_file, "w") as outfile:
        json.dump(cfg, outfile, indent=2)


def compute_coregistration(
    cfg, steps, dem, ref, initial_dh, final_cfg=None, final_json_file=None
):
    """
    Coregister two DSMs together and compute alti differences
    (before and after coregistration).

    This can be view as a two step process:
    - plani rectification computation
    - alti differences computation

    :param cfg: configuration dictionary
    :param dem: xarray.Dataset, dem raster to compare
    :param ref: xarray.Dataset, reference dem raster coregistered with dem
    :param initial_dh: xarray.Dataset, inital alti diff
    :param final_cfg: cfg from a previous run
    :param final_json_file: filename of final_cfg
    :return: coregistered dem and ref rasters + final alti differences
    """
    if "coregistration" in steps:
        (
            coreg_dem,
            coreg_ref,
            final_dh,
        ) = coregistration.coregister_and_compute_alti_diff(cfg, dem, ref)

        # saves results here in case next step fails
        with open(final_json_file, "w") as outfile:
            json.dump(cfg, outfile, indent=2)

    else:
        # If cfg from a previous run, get previous conf
        if (
            final_cfg is not None
            and "plani_results" in final_cfg
            and "alti_results" in final_cfg
        ):
            cfg["plani_results"] = final_cfg["plani_results"]
            cfg["alti_results"] = final_cfg["alti_results"]
            coreg_dem = read_img(
                str(cfg["alti_results"]["rectifiedDSM"]["path"]),
                no_data=(
                    cfg["alti_results"]["rectifiedDSM"]["nodata"]
                    if "nodata" in cfg["alti_results"]["rectifiedDSM"]
                    else None
                ),
            )
            coreg_ref = read_img(
                str(cfg["alti_results"]["rectifiedRef"]["path"]),
                no_data=(
                    cfg["alti_results"]["rectifiedRef"]["nodata"]
                    if "nodata" in cfg["alti_results"]["rectifiedRef"]
                    else None
                ),
            )
            final_dh = read_img(
                str(cfg["alti_results"]["dzMap"]["path"]),
                no_data=cfg["alti_results"]["dzMap"]["nodata"],
            )

        else:
            # Set a default config for following steps from initial DEMs
            # No coregistration done.
            coreg_ref = ref
            coreg_dem = dem
            final_dh = initial_dh
            cfg["plani_results"] = {}
            cfg["plani_results"]["dx"] = {"bias_value": 0, "unit": "m"}
            cfg["plani_results"]["dy"] = {"bias_value": 0, "unit": "m"}
            cfg["alti_results"] = {}
            cfg["alti_results"]["rectifiedDSM"] = copy.deepcopy(cfg["inputDSM"])
            cfg["alti_results"]["rectifiedRef"] = copy.deepcopy(cfg["inputRef"])

            coreg_dem = save_tif(
                coreg_dem,
                os.path.join(
                    cfg["outputDir"], get_out_file_path("coreg_DEM.tif")
                ),
            )
            coreg_ref = save_tif(
                coreg_ref,
                os.path.join(
                    cfg["outputDir"], get_out_file_path("coreg_REF.tif")
                ),
            )
            final_dh = save_tif(
                final_dh,
                os.path.join(
                    cfg["outputDir"], get_out_file_path("final_dh.tif")
                ),
            )
            cfg["alti_results"]["rectifiedDSM"]["path"] = coreg_dem.attrs[
                "ds_file"
            ]
            cfg["alti_results"]["rectifiedRef"]["path"] = coreg_ref.attrs[
                "ds_file"
            ]
            cfg["alti_results"]["rectifiedDSM"]["nb_points"] = coreg_dem[
                "im"
            ].data.size
            cfg["alti_results"]["rectifiedRef"]["nb_points"] = coreg_ref[
                "im"
            ].data.size
            cfg["alti_results"]["rectifiedDSM"][
                "nb_valid_points"
            ] = np.count_nonzero(~np.isnan(coreg_dem["im"].data))
            cfg["alti_results"]["rectifiedRef"][
                "nb_valid_points"
            ] = np.count_nonzero(~np.isnan(coreg_ref["im"].data))
            cfg["alti_results"]["dzMap"] = {
                "path": final_dh.attrs["ds_file"],
                "zunit": coreg_ref.attrs["zunit"].name,
                "nodata": final_dh.coords["no_data"],
                "nb_points": final_dh["im"].data.size,
                "nb_valid_points": np.count_nonzero(
                    ~np.isnan(final_dh["im"].data.size)
                ),
            }

    return coreg_dem, coreg_ref, final_dh


def compute_initialization(config_json):
    """
    Compute demcompare initialization process :
    Configuration copy, checking, create output dir tree
    and initial output content.

    :param config_json: Config json file
    """

    # read the json configuration file
    with open(config_json, "r") as config_json_file:
        cfg = json.load(config_json_file)

    # create output directory
    cfg["outputDir"] = os.path.abspath(cfg["outputDir"])
    initialization.mkdir_p(cfg["outputDir"])

    # copy config into outputDir
    try:
        shutil.copy(
            config_json,
            os.path.join(cfg["outputDir"], os.path.basename(config_json)),
        )
    except shutil.Error:
        # file exists or file is the same
        pass
    except Exception:  # pylint: disable=try-except-raise
        raise

    # checks config
    initialization.check_parameters(cfg)

    # create output tree dirs for each directory
    for directory in get_otd_dirs(cfg["otd"]):
        initialization.mkdir_p(os.path.join(cfg["outputDir"], directory))

    initialization.initialization_plani_opts(cfg)
    initialization.initialization_alti_opts(cfg)
    initialization.initialization_stats_opts(cfg)

    return cfg


def run_tile(json_file, steps=None, display=False):
    """
    DEMcompare execution for a single tile

    :param json_file: Input Json configuration file (mandatory)
    :param steps: Steps to execute (default: all)
    :param display: Choose Plot show or plot save (default).
    """

    # Set steps to default if None
    if steps is None:
        steps = DEFAULT_STEPS

    #
    # Initialization
    #
    cfg = compute_initialization(json_file)
    print(
        (
            "*** DEMcompare : start processing into {} ***".format(
                cfg["outputDir"]
            )
        )
    )
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
        with open(final_json_file, "r") as file:
            final_cfg = json.load(file)

    #
    # Create datasets
    #
    ref, dem = load_dems(
        cfg["inputRef"]["path"],
        cfg["inputDSM"]["path"],
        ref_nodata=(
            cfg["inputRef"]["nodata"] if "nodata" in cfg["inputRef"] else None
        ),
        dem_nodata=(
            cfg["inputDSM"]["nodata"] if "nodata" in cfg["inputDSM"] else None
        ),
        ref_georef=cfg["inputRef"]["georef"],
        dem_georef=cfg["inputDSM"]["georef"],
        ref_zunit=(
            cfg["inputRef"]["zunit"] if "zunit" in cfg["inputRef"] else "m"
        ),
        dem_zunit=(
            cfg["inputDSM"]["zunit"] if "zunit" in cfg["inputDSM"] else "m"
        ),
        load_data=(cfg["roi"] if "roi" in cfg else True),
    )

    #
    # Compute initial dh and save it
    #
    initial_dh = read_img_from_array(
        ref["im"].data - dem["im"].data, from_dataset=dem, no_data=-32768
    )
    initial_dh = save_tif(
        initial_dh,
        os.path.join(cfg["outputDir"], get_out_file_path("initial_dh.tif")),
    )

    stats.dem_diff_plot(
        initial_dh,
        title="DSMs diff without coregistration (REF - DSM)",
        plot_file=os.path.join(
            cfg["outputDir"], get_out_file_path("initial_dem_diff.png")
        ),
        display=False,
    )

    #
    # Coregister both DSMs together and compute final differences
    #
    coreg_dem, coreg_ref, final_dh = compute_coregistration(
        cfg,
        steps,
        dem,
        ref,
        initial_dh,
        final_cfg=final_cfg,
        final_json_file=final_json_file,
    )
    if final_dh is not initial_dh:
        stats.dem_diff_plot(
            final_dh,
            title="DSMs diff with coregistration (REF - DSM)",
            plot_file=os.path.join(
                cfg["outputDir"], get_out_file_path("final_dem_diff.png")
            ),
            display=False,
        )

    #
    # Compute stats
    #
    compute_stats(
        cfg,
        coreg_dem,
        coreg_ref,
        final_dh,
        display=display,
        final_json_file=final_json_file,
    )

    #
    # Compute reports
    #
    compute_report(cfg, steps, coreg_dem, coreg_ref)


def run(json_file, steps=None, display=False):
    """
    DEMcompare main execution for all tiles.
    Call run_tile() function for each tile.

    :param json_file: Input Json configuration file (mandatory)
    :param steps: Steps to execute (default: all)
    :param display: Choose Plot show or plot save (default).
    """

    # Set steps to default if None
    if steps is None:
        steps = DEFAULT_STEPS

    #
    # Initialization
    #
    setup_logging()
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
