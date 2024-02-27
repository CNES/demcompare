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
import os
import traceback

# Third party imports
import argcomplete
import numpy as np
import rasterio
from affine import Affine

from demcompare import run as run_demcompare_on_tile
from demcompare.img_tools import convert_pix_to_coord
from . import log_conf


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
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: INFO. Should be one of "
        "(DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    return parser


def run_tiles(tiles_config, loglevel):  # pylint:disable=too-many-locals
    """
    Call demcompare_tiles's main
    """

    # Logging configuration
    log_conf.setup_logging(default_level=loglevel)

    with open(tiles_config, "r", encoding="utf-8") as json_file:
        dict_config = json.load(json_file)

    # tiles management
    tile_size = dict_config["tile_size"]
    overlap_size = dict_config["overlap_size"]

    # path management
    working_dir = os.getcwd()
    output_dir = working_dir + "/" + dict_config["output_dir"] + "/"

    # datas
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

    image_height = abs(
        int((intersection_roi.top - intersection_roi.bottom) / ref_dem.res[0])
    )
    image_width = abs(
        int((intersection_roi.left - intersection_roi.right) / ref_dem.res[0])
    )

    logging.info("The intersection DEM size is %s row and %s col", image_height, image_width)
    logging.info("The tile size is %s pixel", tile_size)

    nb_tiles_row = (image_height - overlap_size) // (tile_size - overlap_size)
    nb_tiles_col = (image_width - overlap_size) // (tile_size - overlap_size)

    x_2d = np.empty((nb_tiles_row, nb_tiles_col))
    y_2d = np.empty((nb_tiles_row, nb_tiles_col))
    z_2d = np.empty((nb_tiles_row, nb_tiles_col))

    for row in range(nb_tiles_row):
        for col in range(nb_tiles_col):
            top_left_col = col * (tile_size - overlap_size)
            top_left_row = -row * (tile_size - overlap_size)

            bottom_right_col = top_left_col + tile_size
            bottom_right_row = top_left_row - tile_size

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

            saving_dir = (
                output_dir + "row_" + str(row) + "/col_" + str(col) + "/"
            )

            dict_config["output_dir"] = saving_dir

            with open(
                working_dir + "tiles_config.json", "w", encoding="utf-8"
            ) as f:
                json.dump(dict_config, f)

            run_demcompare_on_tile(
                working_dir + "tiles_config.json", loglevel=loglevel
            )

            with open(
                saving_dir + "/coregistration/coregistration_results.json",
                "r",
                encoding="utf-8",
            ) as coreg_results:
                dict_coreg_results = json.load(coreg_results)

            x_2d[row, col] = dict_coreg_results["coregistration_results"]["dx"][
                "total_offset"
            ]
            y_2d[row, col] = dict_coreg_results["coregistration_results"]["dy"][
                "total_offset"
            ]
            # dz directly in altitude unit, offset is directly bias (no pixel)
            z_2d[row, col] = dict_coreg_results["coregistration_results"]["dz"][
                "total_bias_value"
            ]

            print(
                f"The {row * nb_tiles_col + col + 1} out of "
                f"{nb_tiles_row * nb_tiles_col} tiles is complete"
            )

            os.remove(working_dir + "tiles_config.json")

    x_2d = np.flip(x_2d, axis=0)
    y_2d = np.flip(y_2d, axis=0)
    z_2d = np.flip(z_2d, axis=0)

    np.save(output_dir + "coreg_results_x2D.npy", x_2d)
    np.save(output_dir + "coreg_results_y2D.npy", y_2d)
    np.save(output_dir + "coreg_results_z2D.npy", z_2d)


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
