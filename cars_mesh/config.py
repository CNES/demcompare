#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2023 CNES.
#
# This file is part of cars-mesh

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
Main configuration module of cars-mesh tool.
"""

import json
import os

from . import param


def check_general_items(cfg: dict) -> dict:
    """
    Check general items in configuration

    Parameters
    ----------
    cfg: dict
        Configuration dictionary

    Returns
    -------
    cfg: dict
        Configuration dictionary updated
    """

    if "input_path" not in cfg:
        raise ValueError(
            "Configuration dictionary is missing the 'input_path' field."
        )

    if not isinstance(cfg["input_path"], str):
        raise TypeError(
            f"'input_path' is invalid. It should be a string but got "
            f"'{type(cfg['input_path'])}'."
        )

    if os.path.basename(cfg["input_path"]).split(".")[-1] not in (
        param.PCD_FILE_EXTENSIONS + param.MESH_FILE_EXTENSIONS
    ):
        raise ValueError(
            f"'input_path' extension is invalid. "
            f"It should be in "
            f"{param.PCD_FILE_EXTENSIONS + param.MESH_FILE_EXTENSIONS}."
        )

    for key in ["output_dir", "state_machine"]:
        if key not in cfg:
            raise ValueError(
                f"Configuration dictionary should contains a '{key}' key."
            )

    if not isinstance(cfg["state_machine"], list):
        raise TypeError(
            f"State machine key in configuration should be a list of dict,"
            f" each dict having two keys: 'action' and 'method'. "
            f"Found '{type(cfg['state_machine'])}'."
        )

    if "initial_state" not in cfg:
        cfg["initial_state"] = param.INITIAL_STATES[0]
    else:
        if cfg["initial_state"] not in param.INITIAL_STATES:
            raise ValueError(
                f"Initial state is invalid. It should be in"
                f" {param.INITIAL_STATES}."
            )

    return cfg


def check_state_machine(cfg: dict) -> dict:
    """
    Check state machine parameters in configuration

    Parameters
    ----------
    cfg: dict
        Configuration dictionary

    Returns
    -------
    cfg: dict
        Configuration dictionary updated
    """

    if cfg["state_machine"]:
        for k, element in enumerate(cfg["state_machine"]):
            # Action check
            if "action" not in element:
                raise ValueError(
                    f"'action' key is missing in the "
                    f"{k}th element of the state machine list."
                )

            if element["action"] not in param.TRANSITIONS_METHODS:
                raise ValueError(
                    f"Element #{k} of state machine configuration: action "
                    f"'{element['action']}' unknown. It should be in "
                    f"{list(param.TRANSITIONS_METHODS.keys())}."
                )

            # Texture
            # Check that the parameters for rpc, image texture and utm code
            # are given
            if element["action"] == "texture":
                if not cfg["tif_img_path"]:
                    raise ValueError(
                        "If a texturing step is asked, there should be a "
                        "general configuration parameter 'tif_img_path' "
                        "giving the path to the TIF image texture to process."
                    )
                if not cfg["rpc_path"]:
                    raise ValueError(
                        "If a texturing step is asked, there should be a "
                        "general configuration parameter 'rpc_path' giving "
                        "the path to the RPC data of the image texture."
                    )
                if not cfg["utm_code"]:
                    raise ValueError(
                        "If a texturing step is asked, there should be a "
                        "general configuration parameter utm_code' giving "
                        "the UTM code of the input point cloud or mesh for "
                        "coordinate transforming step."
                    )
                if "image_offset" not in cfg:
                    cfg["image_offset"] = None

            # Method check
            if "method" in element:
                # Method specified
                # Check if valid
                if (
                    element["method"]
                    not in param.TRANSITIONS_METHODS[element["action"]]
                ):
                    raise ValueError(
                        f"Element #{k} of state machine configuration: method "
                        f"'{element['method']}' unknown. It should be in"
                        f" {param.TRANSITIONS_METHODS[element['action']]}."
                    )

            else:
                # Method not specified, then select the one by default
                # (the first one in the list)
                element["method"] = param.TRANSITIONS_METHODS[
                    element["action"]
                ][0]

    return cfg


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


def read_config(cfg_path: str) -> dict:
    """
    Read and check if the config path is valid and readable
    and returns cfg if valid

    Parameters
    ----------
    cfg_path: str
        Path to the JSON configuration file

    Return
    ------
    cfg: dict
        Configuration dictionary
    """
    # Check the path validity
    if not isinstance(cfg_path, str):
        raise TypeError(
            f"Configuration path is invalid. It should be a string but got "
            f"'{type(cfg_path)}'."
        )

    # Check the json extension
    if os.path.basename(cfg_path).split(".")[-1] != "json":
        raise ValueError(
            f"Configuration path should be a JSON file with extension '.json'."
            f" Found '{os.path.basename(cfg_path).split('.')[-1]}'."
        )

    # Read JSON file if present (or json.load raises exception)
    with open(cfg_path, "r", encoding="utf-8") as fstream:
        config = json.load(fstream)
        config_dir = os.path.abspath(os.path.dirname(cfg_path))
        # make potential relative paths absolute
        if "input_path" in config:
            config["input_path"] = make_relative_path_absolute(
                config["input_path"], config_dir
            )
        if "rpc_path" in config:
            config["rpc_path"] = make_relative_path_absolute(
                config["rpc_path"], config_dir
            )
        if "tif_img_path" in config:
            config["tif_img_path"] = make_relative_path_absolute(
                config["tif_img_path"], config_dir
            )

    # return updated config with absolute paths for inputs
    return config


def check_config(cfg_path: str) -> dict:
    """
    Global read and check if the config is readable (read_config),
    with valid contents (check_general_items)
    and valid state machine (check_state_machine)

    and returns read cfg at the end if valid

    Parameters
    ----------
    cfg_path: str
        Path to the JSON configuration file

    Return
    ------
    cfg: dict
        Configuration dictionary
    """

    # Read and check config
    cfg = read_config(cfg_path)

    # Check the validity of the content
    cfg = check_general_items(cfg)

    # Check state machine
    cfg = check_state_machine(cfg)

    return cfg


def save_config_file(config_file: str, config: dict):
    """
    Save a json configuration file

    :param config_file: path to a json file
    :type config_file: string
    :param config: configuration json dictionary
    :type config: Dict[str, Any]
    """
    with open(config_file, "w", encoding="utf-8") as file_:
        json.dump(config, file_, indent=2)
