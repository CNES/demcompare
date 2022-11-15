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
This is where high level parameters are checked and default options are set
"""
import copy

# Standard imports
import errno
import json
import logging
import os
import sys
from typing import Any, Dict, Tuple

# Third party imports
import rasterio
from astropy import units as u

from .output_tree_design import get_otd_dirs, get_out_file_path, supported_OTD

# Demcompare imports
from .stats_processing import StatsProcessing


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


def read_config_file(config_file: str) -> Dict[str, Any]:
    """
    Read a demcompare input json config file.
    Relative paths will be made absolute.

    :param config_file: Path to json file
    :type config_file: str

    :returns: The json dictionary read from file
    :rtype: Dict[str, Any]
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


def save_config_file(config_file: str, config: Dict[str, Any]):
    """
    Save a json configuration file

    :param config_file: path to a json file
    :type config_file: string
    :param config: configuration json dictionary
    :type config: Dict[str, Any]
    """
    with open(config_file, "w", encoding="utf-8") as file_:
        json.dump(config, file_, indent=2)


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
    cfg = read_config_file(config_json)

    # Checks input parameters config
    check_input_parameters(cfg)

    if "statistics" in cfg:
        logging.info("Verify statistics configuration")
        cfg_verif = copy.deepcopy(cfg)
        _ = StatsProcessing(cfg=cfg_verif["statistics"])

    # Create output directory and update config
    output_dir = os.path.abspath(cfg["output_dir"])
    cfg["output_dir"] = output_dir
    # Save output_dir parameter in "coregistration" and/or "statistics" dict
    if "coregistration" in cfg:
        cfg["coregistration"]["output_dir"] = output_dir
    if "statistics" in cfg:
        cfg["statistics"]["output_dir"] = output_dir

    # Create output_dir
    mkdir_p(cfg["output_dir"])

    # Save initial config with inputs absolute paths into output_dir
    save_config_file(
        os.path.join(cfg["output_dir"], os.path.basename(config_json)), cfg
    )

    # create output tree dirs for each directory
    for directory in get_otd_dirs(cfg["otd"]):
        mkdir_p(os.path.join(cfg["output_dir"], directory))

    # If defined, force the sampling_source of the
    # coregistration step into the stats step
    if "coregistration" in cfg:
        if "sampling_source" in cfg["coregistration"]:
            if "statistics" in cfg:
                cfg["statistics"]["sampling_source"] = cfg["coregistration"][
                    "sampling_source"
                ]

    return cfg


def check_input_parameters(cfg: Dict[str, Any]):  # noqa: C901
    """
    Checks parameters

    :param cfg: configuration dictionary
    :type cfg: Dict[str, Any]
    """
    input_dems = []
    # If coregistration is present in cfg, boths dems
    # have to be defined
    if "coregistration" in cfg:
        # Verify that both input dems are defined
        if "input_sec" not in cfg:
            raise NameError("ERROR: missing input sec in cfg")
        if "input_ref" not in cfg:
            raise NameError("ERROR: missing input ref in cfg")
        input_dems.append("input_sec")
        input_dems.append("input_ref")

    # If only statistics step is present in cfg,
    # only one dem has to be defined (both is optional)
    elif "statistics" in cfg:
        # Verify that at least one dem is defined
        if "input_ref" not in cfg:
            raise NameError("ERROR: missing input ref in cfg")
        input_dems.append("input_ref")
        # Input_sec is optional
        if "input_sec" in cfg:
            input_dems.append("input_sec")

    for dem in input_dems:
        # Verify and make path absolute
        if "path" not in cfg[dem]:
            raise NameError(f"ERROR: missing paths to {dem}")
        # Verify masks size
        if "classification_layers" in cfg[dem]:
            img_dem = rasterio.open(cfg[dem]["path"])
            for key in cfg[dem]["classification_layers"]:
                if "map_path" in cfg[dem]["classification_layers"][key]:
                    mask_dem = rasterio.open(
                        cfg[dem]["classification_layers"][key]["map_path"]
                    )
                    if img_dem.shape != mask_dem.shape:
                        logging.error(
                            "Dem shape : %s not equal to mask shape : %s",
                            img_dem.shape,
                            mask_dem.shape,
                        )
                        sys.exit(1)
        # Verify z units
        if "zunit" not in cfg[dem]:
            cfg[dem]["zunit"] = "m"
        else:
            try:
                unit = u.Unit(cfg[dem]["zunit"])
            except ValueError as value_error:
                output_msg = cfg[dem]["zunit"]
                raise NameError(
                    f"ERROR: input DSM zunit ({output_msg}) not a "
                    "supported unit"
                ) from value_error
            if unit.physical_type != u.m.physical_type:
                output_msg = cfg[dem]["zunit"]
                raise NameError(
                    f"ERROR: input DSM zunit ({output_msg}) not a lenght unit"
                )
    # check output tree design
    if "otd" in cfg and cfg["otd"] not in supported_OTD:
        otd_name = cfg["otd"]
        raise NameError(
            "ERROR: output tree design set by user"
            f" ({otd_name}) is not supported"
            f" (available options are {supported_OTD})"
        )
    # else
    cfg["otd"] = "default_OTD"


def get_output_files_paths(
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
