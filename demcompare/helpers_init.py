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

# Standard imports
import copy
import json
import logging
import os
from typing import Tuple

# Third party imports
import rasterio
from astropy import units as u

# Demcompare imports
from .dem_processing import DemProcessing
from .internal_typing import ConfigType
from .stats_processing import StatsProcessing


def make_relative_path_absolute(path, directory):
    """
    If path is a valid relative path with respect to directory,
    returns it as an absolute path

    :param path: The relative path
    :type path: string
    :param directory: The directory path should be relative to
    :type directory: string
    :return: os.path.join(directory,path)
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
    :return: The json dictionary read from file with absolute paths
    :rtype: ConfigType
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
    :param config: configuration json dictionary
    :type config: ConfigType
    """
    with open(config_file, "w", encoding="utf-8") as file_:
        json.dump(config, file_, indent=2)


def compute_initialization(config_json: str) -> ConfigType:
    """
    Compute demcompare initialization process :
    Configuration copy, checking,
    and initial output content.

    :param config_json: Config json file name
    :type config_json: str
    :return: demcompare config initialized with default values
    :rtype: ConfigType
    """

    # Read the json configuration file
    # (and update inputs path with absolute path)
    cfg = read_config_file(config_json)

    # Checks input parameters config
    check_input_parameters(cfg)
    # Check statistics configuration by invoking StatsProcessing
    if "statistics" in cfg:
        logging.info("Verify statistics configuration")
        cfg_verif = copy.deepcopy(cfg)
        _ = StatsProcessing(cfg=cfg_verif["statistics"])

    # Create output directory and update config
    output_dir = os.path.abspath(cfg["output_dir"])
    cfg["output_dir"] = output_dir

    # Save output_dir parameter in "coregistration" and/or "statistics" dict
    if "coregistration" in cfg:
        cfg["coregistration"]["output_dir"] = os.path.join(
            cfg["output_dir"], "coregistration"
        )
    if "statistics" in cfg:
        for dem_processing_method in cfg["statistics"]:
            cfg["statistics"][dem_processing_method]["output_dir"] = (
                os.path.join(cfg["output_dir"], "stats", dem_processing_method)
            )

    return cfg


def check_input_parameters(cfg: ConfigType):  # noqa: C901
    """
    Checks parameters

    :param cfg: configuration dictionary
    :type cfg: ConfigType
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
        # Coregistration without statistics is allowed

    # If only statistics step (without coreg) is present in cfg, two cases:
    # 1. stats on one dem on input_ref only (input_sec only is not allowed)
    # 2. stats on two dem diff (without coreg) with input_ref and input_sec
    elif "statistics" in cfg:
        # Verify that at least one dem is defined in input_ref
        if "input_ref" not in cfg:
            raise NameError("ERROR: missing input ref in cfg")
        input_dems.append("input_ref")
        # Input_sec is optional, case 2 if present, case 1 otherwise.
        if "input_sec" in cfg:
            input_dems.append("input_sec")
    else:
        raise NameError("ERROR: missing configuration steps")

    # Check input_dems paths, masks, units
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
                        raise ValueError(
                            f"Dem shape : {img_dem.shape} not equal "
                            "to mask shape : {mask_dem.shape}"
                        )

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
    # Check report config
    if "report" in cfg:
        # Supported for now: default (-> sphinx) and sphinx
        if cfg["report"] == "default":
            cfg["report"] = "sphinx"
        elif cfg["report"] != "sphinx":
            report_name = cfg["report"]
            raise NameError(
                f"ERROR: {report_name} is not supported,"
                "report type must be sphinx only for now"
            )

    check_dem_processing_methods(cfg)

    check_curvature_slope(cfg)


def check_dem_processing_methods(cfg: ConfigType):
    """
    Checks that the DEM processing methods
    in the config are correct.

    :param cfg: configuration dictionary
    :type cfg: ConfigType
    """

    if "statistics" in cfg:
        for dem_processing_method in cfg["statistics"]:
            if (
                dem_processing_method
                not in DemProcessing.available_dem_processing_methods
            ):
                raise NameError(
                    f"DEM processing method: {dem_processing_method}"
                    "is not correct"
                )


def check_curvature_slope(cfg: ConfigType):
    """
    Checks that there is no
    DEM processing method '..-curvature'
    with 'Slope0' as 'classification_layers'

    :param cfg: configuration dictionary
    :type cfg: ConfigType
    """

    if "statistics" in cfg:
        if "ref-curvature" in cfg["statistics"]:
            if "classification_layers" in cfg["statistics"]["ref-curvature"]:
                if (
                    "Slope0"
                    in cfg["statistics"]["ref-curvature"][
                        "classification_layers"
                    ]
                ):
                    raise NameError(
                        "The DEM processing method: 'ref-curvature'",
                        "cannot have 'Slope0' as 'classification_layers'",
                    )
        if "sec-curvature" in cfg["statistics"]:
            if "classification_layers" in cfg["statistics"]["sec-curvature"]:
                if (
                    "Slope0"
                    in cfg["statistics"]["sec-curvature"][
                        "classification_layers"
                    ]
                ):
                    raise NameError(
                        "The DEM processing method: 'sec-curvature'",
                        "cannot have 'Slope0' as 'classification_layers'",
                    )


def get_output_files_paths(
    output_dir: str, dir_name: str, file_name: str
) -> Tuple[str, str, str, str, str, str, str, str]:
    """
    Return the paths of the output global files:
    - dem.tif
    - dem.png
    - dem_cdf.tif and dem_cdf.csv
    - dem_pdf.tif and dem_pdf.csv

    :param output_dir: output_dir
    :type output_dir: str
    :param dir_name: name of the subdirectory
    :type dir_name: str
    :param file_name: name of the files
    :type file_name: str
    :return: Output paths
    :rtype: Tuple[str, str, str, str, str, str, str, str]
    """
    # Compute and save image tif and image plot png
    dem_path = os.path.join(output_dir, "stats", dir_name, file_name + ".tif")
    plot_file_path = os.path.join(
        output_dir, "stats", dir_name, file_name + "_snapshot.png"
    )
    plot_path_cdf = os.path.join(
        output_dir, "stats", dir_name, file_name + "_cdf.png"
    )
    csv_path_cdf = os.path.join(
        output_dir, "stats", dir_name, file_name + "_cdf.csv"
    )
    plot_path_pdf = os.path.join(
        output_dir, "stats", dir_name, file_name + "_pdf.png"
    )
    csv_path_pdf = os.path.join(
        output_dir, "stats", dir_name, file_name + "_pdf.csv"
    )
    plot_path_svf = os.path.join(
        output_dir, "stats", dir_name, file_name + "_svf.png"
    )
    plot_path_hillshade = os.path.join(
        output_dir, "stats", dir_name, file_name + "_hillshade.png"
    )
    return (
        dem_path,  # pylint:disable=duplicate-code
        plot_file_path,  # pylint:disable=duplicate-code
        plot_path_cdf,  # pylint:disable=duplicate-code
        csv_path_cdf,  # pylint:disable=duplicate-code
        plot_path_pdf,  # pylint:disable=duplicate-code
        csv_path_pdf,  # pylint:disable=duplicate-code
        plot_path_svf,  # pylint:disable=duplicate-code
        plot_path_hillshade,  # pylint:disable=duplicate-code
    )
