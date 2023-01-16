#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 CNES.
#
# This file is part of mesh3d
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
Main Reconstruct module of mesh3d tool.
"""

import json
import logging
import os
import shutil

from loguru import logger

from . import param
from .state_machine import Mesh3DMachine
from .tools.handlers import Mesh, read_input_path


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


def check_config(cfg_path: str) -> dict:
    """
    Check if the config is valid and readable

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

    if os.path.basename(cfg_path).split(".")[-1] != "json":
        raise ValueError(
            f"Configuration path should be a JSON file with extension '.json'."
            f" Found '{os.path.basename(cfg_path).split('.')[-1]}'."
        )

    # Read JSON file
    with open(cfg_path, "r", encoding="utf-8") as cfg_file:
        cfg = json.load(cfg_file)

    # Check the validity of the content
    cfg = check_general_items(cfg)

    # Check state machine
    cfg = check_state_machine(cfg)

    return cfg


def run(mesh3d_machine: Mesh3DMachine, cfg: dict) -> Mesh:
    """
    Run the state machine

    Parameters
    ----------
    mesh3d_machine: Mesh3DMachine
        Mesh 3D state machine model
    cfg: dict
        Configuration dictionary

    Returns
    -------
    mesh: Mesh
        Mesh object
    """

    if not cfg["state_machine"]:
        # There is no step given to the state machine
        logger.warning("State machine is empty. Returning the initial data.")

    else:
        logger.debug(f"Initial state: {mesh3d_machine.initial_state}")

        # Check transitions' validity
        mesh3d_machine.check_transitions(cfg)

        # Browse user defined steps and execute them
        for k, step in enumerate(cfg["state_machine"]):
            # Logger
            logger.info(
                f"Step #{k + 1}: {step['action']} with {step['method']} method"
            )

            # Run action
            mesh3d_machine.run(step, cfg)

            # (Optional) Save intermediate results to disk if asked
            if "save_output" in step:
                if step["save_output"]:
                    # Create directory to save intermediate results
                    intermediate_folder = os.path.join(
                        cfg["output_dir"], "intermediate_results"
                    )
                    os.makedirs(intermediate_folder, exist_ok=True)
                    if k == 0 and os.listdir(intermediate_folder):
                        logger.warning(
                            f"Directory '{intermediate_folder}' is not empty. "
                            f"Some files might be overwritten."
                        )

                    # Save intermediates results
                    intermediate_filepath = os.path.join(
                        intermediate_folder,
                        f"{(str(k+1)).zfill(2)}"
                        f"_{step['action']}"
                        f"_{step['method']}",
                    )
                    if mesh3d_machine.mesh_data.df is not None:
                        # Mesh serialisation
                        extension = "ply"
                        mesh3d_machine.mesh_data.serialize(
                            filepath=intermediate_filepath + "." + extension,
                            extension=extension,
                        )
                    else:
                        # Point cloud serialisation
                        extension = "laz"
                        mesh3d_machine.mesh_data.pcd.serialize(
                            filepath=intermediate_filepath + "." + extension,
                            extension=extension,
                        )

                    # Logger debug
                    logger.debug(
                        f"Step #{k + 1}: Results saved in "
                        f"'{intermediate_filepath  + '.' + extension}'."
                    )

    return mesh3d_machine.mesh_data


def main(cfg_path: str) -> None:
    """
    Main function to apply mesh 3d pipeline

    Parameters
    ----------
    cfg_path: str
        Path to the JSON configuration file
    """
    # To avoid having a logger INFO for each state machine step
    logging.getLogger("transitions").setLevel(logging.WARNING)

    # Check the validity of the config path
    cfg = check_config(cfg_path)

    # Copy the configuration file in the output dir
    os.makedirs(cfg["output_dir"], exist_ok=True)
    shutil.copy(
        cfg_path,
        os.path.join(
            os.path.join(cfg["output_dir"], os.path.basename(cfg_path))
        ),
    )

    # Write logs to disk
    logger.add(sink=os.path.join(cfg["output_dir"], "{time}_logs.txt"))
    logger.info("Configuration file checked.")

    # Read input data
    mesh = read_input_path(cfg["input_path"])

    # Init state machine model
    mesh3d_machine = Mesh3DMachine(
        mesh_data=mesh, initial_state=cfg["initial_state"]
    )

    # Run the pipeline according to the user configuration
    out_mesh = run(mesh3d_machine, cfg)

    # Serialize data
    # Check if user specified an output name
    # otherwise assign a default one
    if "output_name" not in cfg:
        cfg["output_name"] = "output_mesh3d"  # default output name
    # If data is only a point cloud, prefer output extension 'laz'
    # otherwise use 'ply'
    extension = "ply" if out_mesh.df is not None else "laz"
    out_filename = cfg["output_name"] + "." + extension

    if out_mesh.df is not None:
        # Mesh Serialisation
        out_mesh.serialize(
            filepath=os.path.join(cfg["output_dir"], out_filename),
            extension=extension,
        )
        logger.info(f"Mesh serialized as a '{extension}' file")
    else:
        # Point Cloud Serialisation
        out_mesh.pcd.serialize(
            filepath=os.path.join(cfg["output_dir"], out_filename),
            extension=extension,
        )
        logger.info(f"Point cloud serialized as a '{extension}' file")
