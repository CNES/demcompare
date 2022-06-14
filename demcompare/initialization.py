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
Init part of sec_compare
This is where high level parameters are checked and default options are set
TODO: move all init parts of __init__ here for consistency
"""

# Standard imports
import copy
import errno
import json
import logging
import os
from typing import Any, Dict, List, Tuple

# Third party imports
import numpy as np
from astropy import units as u

# Demcompare imports
from .dem_tools import load_dem
from .output_tree_design import get_out_file_path, supported_OTD

# Declare a configuration json type for type hinting
ConfigType = Dict[str, Any]


def mkdir_p(path):
    """
    Create a directory without complaining if it already exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # requires Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def make_relative_path_absolute(path, directory):
    """
    If path is a valid relative path with respect to directory,
    returns it as an absolute path

    :param path: The relative path
    :type path: string
    :param directory: The directory path should be relative to
    :type directory: string
    :returns: os.path.join(directory,path)
        if path is a valid relative path form directory, else path
    :rtype: string
    """
    out = path
    if not os.path.isabs(path):
        abspath = os.path.join(directory, path)
        if os.path.exists(abspath):
            out = abspath
    return out


def read_config_file(config_file: str) -> ConfigType:
    """
    Read a demcompare input json config file.
    Relative paths will be made absolute.

    :param config_file: Path to json file
    :type config_file: str

    :returns: The json dictionary read from file
    :rtype: dict
    """
    with open(config_file, "r", encoding="utf-8") as _fstream:
        # Load json file
        config = json.load(_fstream)
        config_dir = os.path.abspath(os.path.dirname(config_file))
        # make potential relative paths absolute
        if "input_ref" in config:
            config["input_ref"]["path"] = make_relative_path_absolute(
                config["input_ref"]["path"], config_dir
            )
            if "classification_layers" in config["input_ref"]:
                for _, classif_cfg in config["input_ref"][
                    "classification_layers"
                ].items():
                    classif_cfg["map_path"] = make_relative_path_absolute(
                        classif_cfg["map_path"], config_dir
                    )
        if "input_sec" in config:
            config["input_sec"]["path"] = make_relative_path_absolute(
                config["input_sec"]["path"], config_dir
            )
            if "classification_layers" in config["input_sec"]:
                for _, classif_cfg in config["input_sec"][
                    "classification_layers"
                ].items():
                    classif_cfg["map_path"] = make_relative_path_absolute(
                        classif_cfg["map_path"], config_dir
                    )
    return config


def save_config_file(config_file: str, config: ConfigType):
    """
    Save a json configuration file

    :param config_file: path to a json file
    :type config_file: string
    :param config_file: configuration json dictionary
    :type config_file: dict
    """
    with open(config_file, "w", encoding="utf-8") as file_:
        json.dump(config, file_, indent=2)


def check_input_parameters(cfg: ConfigType):  # noqa: C901
    """
    Checks parameters

    :param cfg: configuration dictionary
    :type cfg: Dict
    """
    # verify that input files are defined
    if "input_sec" not in cfg or "input_ref" not in cfg:
        raise NameError("ERROR: missing input images description")

    # verify if input files are correctly defined :
    if "path" not in cfg["input_sec"] or "path" not in cfg["input_ref"]:
        raise NameError("ERROR: missing paths to input images")
    cfg["input_sec"]["path"] = os.path.abspath(str(cfg["input_sec"]["path"]))
    cfg["input_ref"]["path"] = os.path.abspath(str(cfg["input_ref"]["path"]))

    # verify z units
    if "zunit" not in cfg["input_sec"]:
        cfg["input_sec"]["zunit"] = "m"
    else:
        try:
            unit = u.Unit(cfg["input_sec"]["zunit"])
        except ValueError as value_error:
            raise NameError(
                "ERROR: input DSM zunit ({}) not a supported unit".format(
                    cfg["input_sec"]["zunit"]
                )
            ) from value_error
        if unit.physical_type != u.m.physical_type:
            raise NameError(
                "ERROR: input DSM zunit ({}) not a lenght unit".format(
                    cfg["input_sec"]["zunit"]
                )
            )
    if "zunit" not in cfg["input_ref"]:
        cfg["input_ref"]["zunit"] = "m"
    else:
        try:
            unit = u.Unit(cfg["input_ref"]["zunit"])
        except ValueError as value_error:
            raise NameError(
                "ERROR: input Ref zunit ({}) not a supported unit".format(
                    cfg["input_ref"]["zunit"]
                )
            ) from value_error
        if unit.physical_type != u.m.physical_type:
            raise NameError(
                "ERROR: input Ref zunit ({}) not a lenght unit".format(
                    cfg["input_ref"]["zunit"]
                )
            )

    # check ref:
    if "geoid" in cfg["input_sec"] or "geoid" in cfg["input_ref"]:
        logging.warning(
            "WARNING : geoid option is deprecated. \
            Use georef keyword now with EGM96 or WGS84 value"
        )
    # what we do below is just in case someone used georef as geoid was used...
    if "georef" in cfg["input_sec"]:
        if cfg["input_sec"]["georef"] is True:
            cfg["input_sec"]["georef"] = "EGM96"
        else:
            if cfg["input_sec"]["georef"] is False:
                cfg["input_sec"]["georef"] = "WGS84"
    else:
        cfg["input_sec"]["georef"] = "WGS84"
    if "georef" in cfg["input_ref"]:
        if cfg["input_ref"]["georef"] is True:
            cfg["input_ref"]["georef"] = "EGM96"
        else:
            if cfg["input_ref"]["georef"] is False:
                cfg["input_ref"]["georef"] = "WGS84"
    else:
        cfg["input_ref"]["georef"] = "WGS84"

    # check output tree design
    if "otd" in cfg and cfg["otd"] not in supported_OTD:
        raise NameError(
            "ERROR: output tree design set by user ({}) is not supported"
            " (available options are {})".format(cfg["otd"], supported_OTD)
        )
    # else
    cfg["otd"] = "default_OTD"


def get_tile_dir(cfg: Dict, col_1: int, row_1: int, width: int, height: int):
    """
    Get the name of a tile directory

    :param cfg: Input demcompare configuration
    :type cfg: Dict
    :param col_1: value of tile first column
    :type col_1: int
    :param row_1: value of tile first row
    :type row_1: int
    :param width: width of tile
    :type width: int
    :param height: height of tile
    :type height: int
    :return: None
    """
    max_digit_row = 0
    max_digit_col = 0
    if "max_digit_tile_row" in cfg:
        max_digit_row = cfg["max_digit_tile_row"]
    if "max_digit_tile_col" in cfg:
        max_digit_col = cfg["max_digit_tile_col"]
    return os.path.join(
        cfg["output_dir"],
        "tiles",
        "row_{:0{}}_height_{}".format(row_1, max_digit_row, height),
        "col_{:0{}}_width_{}".format(col_1, max_digit_col, width),
    )


# TODO: to be modified by the refacto script demcompare
def adjust_tile_size(image_size: int, tile_size: int) -> Tuple[int, int]:
    """
    Adjust the size of the tiles.

    :param image_size: Input demcompare configuration
    :type image_size: Dict
    :param tile_size: value of tile first column
    :type tile_size: int
    :return: tile_w, tile_h
    :rtype: int, int
    """
    tile_w = min(image_size["w"], tile_size)  # tile width
    ntx = int(np.round(float(image_size["w"]) / tile_w))
    # ceil so that, if needed, the last tile is slightly smaller
    tile_w = int(np.ceil(float(image_size["w"]) / ntx))

    tile_h = min(image_size["h"], tile_size)  # tile height
    nty = int(np.round(float(image_size["h"]) / tile_h))
    tile_h = int(np.ceil(float(image_size["h"]) / nty))

    logging.info(("tile size: {} {}".format(tile_w, tile_h)))

    return tile_w, tile_h


# TODO: to be modified by the refacto script demcompare
def compute_tiles_coordinates(
    roi: dict, tile_width: int, tile_height: int
) -> List[Tuple[int, int, int, int]]:
    """
    Compute tiles coordinates

    :param roi: Region of interest
    :type roi: dict
    :param tile_width: width of tile
    :type tile_width: int
    :param tile_height:  height of tile
    :type tile_height: int
    :return: out
    :rtype: List[Tuple[int, int, int, int]]
    """
    out = []
    for row in np.arange(roi["y"], roi["y"] + roi["h"], tile_height):
        height = min(tile_height, roi["y"] + roi["h"] - row)
        for col in np.arange(roi["x"], roi["x"] + roi["w"], tile_width):
            width = min(tile_width, roi["x"] + roi["w"] - col)
            out.append((col, row, width, height))

    return out


# TODO: to be modified by the refacto script demcompare
def divide_images(cfg: ConfigType):
    """
    List the tiles to process and prepare their output directories structures.

    :param cfg: cfg
    :type cfg: dict
    :return: tiles, contains the image coordinates
            and the output directory path of a tile.
    :rtype: List[dict]
    """

    # compute biggest roi
    dem = load_dem(
        cfg["input_sec"]["path"],
        input_roi=(cfg["roi"] if "roi" in cfg else False),
    )

    sizes = {"w": dem["image"].data.shape[1], "h": dem["image"].data.shape[0]}
    roi = {
        "x": cfg["roi"]["x"] if "roi" in cfg else 0,
        "y": cfg["roi"]["y"] if "roi" in cfg else 0,
        "w": dem["image"].data.shape[1],
        "h": dem["image"].data.shape[0],
    }

    # list tiles coordinates
    tile_size_w, tile_size_h = adjust_tile_size(sizes, cfg["tile_size"])
    tiles_coords = compute_tiles_coordinates(roi, tile_size_w, tile_size_h)
    if "max_digit_tile_row" not in cfg:
        cfg["max_digit_tile_row"] = len(
            str(tiles_coords[len(tiles_coords) - 1][0])
        )
    if "max_digit_tile_col" not in cfg:
        cfg["max_digit_tile_col"] = len(
            str(tiles_coords[len(tiles_coords) - 1][1])
        )

    # build a tile dictionary for all non-masked tiles and store them in a list
    tiles = []
    for coords in tiles_coords:
        tile = {}
        col, row, width, height = coords
        tile["dir"] = get_tile_dir(cfg, col, row, width, height)
        tile["coordinates"] = coords
        tiles.append(tile)

    # make tiles directories and store json configuration
    for tile in tiles:
        mkdir_p(tile["dir"])

        # save a json dump of the tile configuration
        tile_cfg = copy.deepcopy(cfg)
        col, row, width, height = tile["coordinates"]
        tile_cfg["roi"] = {"x": col, "y": row, "w": width, "h": height}
        tile_cfg["output_dir"] = tile["dir"]

        tile_json = os.path.join(tile["dir"], "config.json")
        tile["json"] = tile_json

        save_config_file(tile_json, tile_cfg)

    # Write the list of json files to output_dir/tiles.txt
    with open(
        os.path.join(cfg["output_dir"], "tiles.txt"), "w", encoding="utf8"
    ) as tile_file:
        for tile in tiles:
            tile_file.write(tile["json"] + os.linesep)

    return tiles


def compute_output_files_paths(
    output_dir, name
) -> Tuple[str, str, str, str, str, str]:
    """
    Return the paths of the output global files:
    - dem.png
    - dem.tiff
    - dem_cdf.tiff and dem_cdf.csv
    - dem_pdf.tiff and dem_pdf.csv

    :param output_dir: output_dir
    :type output_dir: str
    :param name: name
    :type name: str
    :return: Output paths
    :rtype: Tuple[str, str, str, str, str, str]
    """
    # Compute and save image tiff and image plot png
    dem_path = os.path.join(output_dir, get_out_file_path(name + ".tif"))
    plot_file_path = os.path.join(output_dir, get_out_file_path(name + ".png"))
    plot_path_cdf = os.path.join(
        output_dir, get_out_file_path(name + "_cdf.png")
    )
    csv_path_cdf = os.path.join(
        output_dir, get_out_file_path(name + "_cdf.csv")
    )
    plot_path_pdf = os.path.join(
        output_dir, get_out_file_path(name + "_pdf.png")
    )
    csv_path_pdf = os.path.join(
        output_dir, get_out_file_path(name + "_pdf.csv")
    )
    return (
        dem_path,
        plot_file_path,
        plot_path_cdf,
        csv_path_cdf,
        plot_path_pdf,
        csv_path_pdf,
    )
