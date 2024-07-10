#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module contains a wrapper for performing tiling on datas
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import traceback
from typing import Callable, Tuple

# Third party imports
import argcomplete
import numpy as np
import rasterio
from affine import Affine

from demcompare import log_conf
from demcompare import run as run_demcompare_on_tile
from demcompare.img_tools import convert_pix_to_coord


def get_parser():
    """
    ArgumentParser for demcompare_tiles

    :return: parser
    """
    parser = argparse.ArgumentParser(
        description="Compare Digital Elevation Models by tiling",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "tiles_config",
        metavar="config.json",
        help=(
            "path to a json file containing the paths to "
            "input and output files, the tiles parameters "
            "and the algorithm parameters"
        ),
    )

    parser.add_argument(
        "--loglevel",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: INFO. Should be one of "
        "(DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    return parser


def process_tile(args):
    """
    Function that uses multiprocessing to run `demcompare`
    on multiple tiles concurrently.

    :param args: arguments list to compute process_tile with multiprocessing,
        see list below
    :type args: list

        :param row: Row index of the tile
        :type row: int
        :param col: Column index of the tile
        :type col: int
        :param width: Width of the tile
        :type width: int
        :param height: Height of the tile
        :type height: int
        :param overlap_size: overlap_size between two tiles
        :type overlap_size: int
        :param new_geotransform: geotransform's intersection of two tiles
        :type new_geotransform: affine
        :param output_dir: output directory for one tile
        :type output_dir: str
        :param dict_config: Tile's demcompare configuration
        :type dict_config: dict
        :param loglevel: log level
        :type loglevel: str
    """

    (
        row,
        col,
        width,
        height,
        overlap_size,
        new_geotransform,
        output_dir,
        dict_config,
        loglevel,
    ) = args

    # Get tile in DEM
    top_left_col = col * (width - overlap_size)
    top_left_row = -row * (height - overlap_size)

    bottom_right_col = top_left_col + width
    bottom_right_row = top_left_row - height

    left_point = convert_pix_to_coord(
        new_geotransform, top_left_row, top_left_col
    )
    right_point = convert_pix_to_coord(
        new_geotransform, bottom_right_row, bottom_right_col
    )

    roi = {
        "left": float(left_point[0]),
        "bottom": float(left_point[1]),
        "right": float(right_point[0]),
        "top": float(right_point[1]),
    }

    dict_config["input_ref"]["roi"] = roi
    dict_config["input_sec"]["roi"] = roi

    saving_dir = os.path.join(output_dir, f"row_{row}/col_{col}/")
    os.makedirs(saving_dir, exist_ok=True)

    dict_config["output_dir"] = saving_dir
    config = os.path.join(output_dir, f"with_roi_{row}_{col}.json")

    # Create new config with roi
    with open(config, "w", encoding="utf-8") as f:
        json.dump(dict_config, f)

    try:
        # run demcompare
        run_demcompare_on_tile(config, loglevel=loglevel)

    # If gradient function doesn't work on tile
    except ValueError:
        logging.info(
            "Tile (%s, %s) is too small, NaN values are returned", row, col
        )
        shutil.rmtree(saving_dir)

    # remove tile config from disk
    os.remove(config)


def verify_config(dict_config_tiling: dict) -> Tuple[int, int, int, int]:
    """
    Functions that verify tiling configuration

    :param dict_config_tiling: dictionary containing the tiles parameters
    :type dict_config_tiling: dict
    :return: height, width, overlap_size, nb_cpu
    :rtype: Tuple[int, int, int, int]
    """

    # Function to validate each parameter
    def validate_param(
        param_name: str, condition: Callable, error_message: str
    ) -> int:
        """
        Validate parameter from tile_configuration
        :param param_name: a tile's parameter
        :type param_name: str
        :param condition: a lambda function describing the condition
        :type condition: Callable
        :param error_message: Error message for user
        :type error_message: str

        :return: The parameter value
        :rtype: int
        """
        if param_name in dict_config_tiling and condition(
            dict_config_tiling[param_name]
        ):
            return dict_config_tiling[param_name]

        raise ValueError(error_message)

    # Define validation conditions
    conditions = {
        "height": lambda x: isinstance(x, int) and x > 0,
        "width": lambda x: isinstance(x, int) and x > 0,
        "overlap": lambda x: isinstance(x, int) and x >= 0,
        "nb_cpu": lambda x: isinstance(x, int) and x > 0,
    }

    # Validate parameters
    height = validate_param(
        "height", conditions["height"], "Height is not consistent"
    )
    width = validate_param(
        "width", conditions["width"], "Width is not consistent"
    )
    overlap_size = validate_param(
        "overlap", conditions["overlap"], "Overlap is not consistent"
    )

    # Get actual CPU usage from the environment to automate the default pipeline
    # and validate CPU load.
    if "nb_cpu" not in dict_config_tiling:
        nb_cpu = len(os.sched_getaffinity(0))
    else:
        nb_cpu = validate_param(
            "nb_cpu", conditions["nb_cpu"], "Number of CPUs is incorrect"
        )
    if nb_cpu > len(os.sched_getaffinity(0)):
        raise ValueError(
            "Number of CPUs in the config is more than available CPUs"
        )

    return height, width, overlap_size, nb_cpu


def get_coreg_results(
    coreg_results: dict,
) -> Tuple[float, float, float, float, float]:
    """
    Get coregistration results for one tile

    :param coreg_results: Coregistration dictionary results
    :type coreg_results: dict
    :return: x, y and z shifts, percentage of valid data (ref, sec)
    :rtype: float, float, float, float, float
    """

    x = coreg_results["coregistration_results"]["dx"]["total_offset"]
    y = coreg_results["coregistration_results"]["dy"]["total_offset"]
    z = coreg_results["coregistration_results"]["dz"]["total_bias_value"]

    nb_valid_pts_ref = coreg_results["coregistration_results"][
        "reproj_coreg_ref"
    ]["percentage_valid_points"]
    nb_valid_pts_sec = coreg_results["coregistration_results"][
        "reproj_coreg_sec"
    ]["percentage_valid_points"]

    return x, y, z, nb_valid_pts_ref, nb_valid_pts_sec


def run_tiles(tiles_config, loglevel):  # pylint:disable=too-many-locals
    """
    Call demcompare_tiles's main
    """
    # Logging configuration
    log_conf.setup_logging(default_level=loglevel)

    with open(tiles_config, "r", encoding="utf-8") as json_file:
        dict_config = json.load(json_file)

    # Verify config and get tiles management
    height, width, overlap_size, nb_cpu = verify_config(dict_config["tiling"])

    # Path management
    output_dir = os.path.abspath(dict_config["output_dir"])

    dict_config["input_ref"]["path"] = os.path.abspath(
        dict_config["input_ref"]["path"]
    )
    dict_config["input_sec"]["path"] = os.path.abspath(
        dict_config["input_sec"]["path"]
    )
    dict_config["output_dir"] = output_dir

    # Create output_dir from updated absolute path
    os.makedirs(dict_config["output_dir"], exist_ok=True)

    ref_dem = rasterio.open(dict_config["input_ref"]["path"])
    sec_dem = rasterio.open(dict_config["input_sec"]["path"])

    # Get DEM intersection
    transformed_sec_bounds = rasterio.warp.transform_bounds(
        sec_dem.crs,
        sec_dem.crs,
        sec_dem.bounds.left,
        sec_dem.bounds.bottom,
        sec_dem.bounds.right,
        sec_dem.bounds.top,
    )

    transformed_ref_bounds = rasterio.warp.transform_bounds(
        ref_dem.crs,
        sec_dem.crs,
        ref_dem.bounds.left,
        ref_dem.bounds.bottom,
        ref_dem.bounds.right,
        ref_dem.bounds.top,
    )

    if rasterio.coords.disjoint_bounds(
        transformed_sec_bounds, transformed_ref_bounds
    ):
        raise NameError("ERROR: ROIs do not intersect")

    intersection_roi = rasterio.coords.BoundingBox(
        max(transformed_sec_bounds[0], transformed_ref_bounds[0]),
        max(transformed_sec_bounds[1], transformed_ref_bounds[1]),
        min(transformed_sec_bounds[2], transformed_ref_bounds[2]),
        min(transformed_sec_bounds[3], transformed_ref_bounds[3]),
    )

    # Working on intersection
    new_geotransform = list(
        Affine(
            ref_dem.res[0],
            0.0,
            intersection_roi.left,
            0.0,
            -ref_dem.res[1],
            intersection_roi.bottom,
        ).to_gdal()
    )

    # Instance of tiles parameters
    image_height = abs(
        int((intersection_roi.top - intersection_roi.bottom) / ref_dem.res[0])
    )
    image_width = abs(
        int((intersection_roi.left - intersection_roi.right) / ref_dem.res[0])
    )

    logging.info(
        "The intersection DEM size is %s row and %s col",
        image_height,
        image_width,
    )
    logging.info("The tile size is %s row %s col", height, width)

    nb_tiles_row = (image_height - overlap_size) // (height - overlap_size)
    if (image_height - overlap_size) % (height - overlap_size) != 0:
        nb_tiles_row += 1

    nb_tiles_col = (image_width - overlap_size) // (width - overlap_size)
    if (image_width - overlap_size) % (width - overlap_size) != 0:
        nb_tiles_col += 1

    logging.info(
        "There are %s tiles in columns and %s tiles in rows",
        nb_tiles_col,
        nb_tiles_row,
    )

    tasks = [
        (
            row,
            col,
            width,
            height,
            overlap_size,
            new_geotransform,
            output_dir,
            dict_config,
            loglevel,
        )
        for row in range(nb_tiles_row)
        for col in range(nb_tiles_col)
    ]

    with mp.Pool(processes=nb_cpu) as pool:
        _ = pool.map(process_tile, tasks)

    # Compute matrix to store dx, dy, dz and valid points percentage
    x_2d = np.full((nb_tiles_row, nb_tiles_col), np.nan)
    y_2d = np.full((nb_tiles_row, nb_tiles_col), np.nan)
    z_2d = np.full((nb_tiles_row, nb_tiles_col), np.nan)
    percentage_valid_points = np.full((2, nb_tiles_row, nb_tiles_col), np.nan)

    for row in range(nb_tiles_row):
        for col in range(nb_tiles_col):

            # Sometimes demcompare is not robust and the directory is removed
            if os.path.isdir(output_dir + f"/row_{row}/col_{col}"):
                with open(
                    output_dir + f"/row_{row}/col_{col}/coregistration/"
                    f"coregistration_results.json",
                    "r",
                    encoding="utf-8",
                ) as json_file:
                    coreg_results = json.load(json_file)

                x, y, z, valid_point_ref, valid_point_sec = get_coreg_results(
                    coreg_results
                )

                x_2d[row, col] = x
                y_2d[row, col] = y
                z_2d[row, col] = z
                percentage_valid_points[0, row, col] = valid_point_ref
                percentage_valid_points[1, row, col] = valid_point_sec

    np.save(os.path.join(output_dir, "coreg_results_x2D.npy"), x_2d)
    np.save(os.path.join(output_dir, "coreg_results_y2D.npy"), y_2d)
    np.save(os.path.join(output_dir, "coreg_results_z2D.npy"), z_2d)
    np.save(
        os.path.join(output_dir, "percentage_valid_points.npy"),
        percentage_valid_points,
    )


def main():
    """
    Call demcompare-tile's main
    """

    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    try:
        run_tiles(args.tiles_config, args.loglevel)

    except Exception:  # pylint: disable=broad-except
        logging.error(" Demcompare %s", traceback.format_exc())


if __name__ == "__main__":
    main()
