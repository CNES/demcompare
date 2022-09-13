#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Chloe Thenoz (Magellium), Lisa Vo Thanh (Magellium).
#
# This file is part of mesh_3d
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
Main module.
DOC: describe the module aim !
"""

import json
import logging
import os

from loguru import logger

from . import param
from .state_machine import Mesh3DMachine

# from .tools.point_cloud_io import deserialize_point_cloud, serialize_point_cloud
# from .tools.mesh_io import deserialize_mesh, serialize_mesh
from .tools.handlers import Mesh, PointCloud


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
            f"Configuration path is invalid. It should be a string but got '{type(cfg_path)}'."
        )

    if os.path.basename(cfg_path).split(".")[-1] != "json":
        raise ValueError(
            f"Configuration path should be a JSON file with extension '.json'. "
            f"Found '{os.path.basename(cfg_path).split('.')[-1]}'."
        )

    # Read JSON file
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # Check the validity of the content
    if "input_path" not in cfg:
        raise ValueError(
            f"Configuration dictionary is missing the 'input_path' field."
        )
    else:
        if not isinstance(cfg["input_path"], str):
            raise TypeError(
                f"'input_path' is invalid. It should be a string but got '{type(cfg['input_path'])}'."
            )
        if os.path.basename(cfg["input_path"]).split(".")[-1] not in (
            param.PCD_FILE_EXTENSIONS + param.MESH_FILE_EXTENSIONS
        ):
            raise ValueError(
                f"'input_path' extension is invalid. "
                f"It should be in {param.PCD_FILE_EXTENSIONS + param.MESH_FILE_EXTENSIONS}."
            )

    for key in ["output_dir", "state_machine"]:
        if key not in cfg:
            raise ValueError(
                f"Configuration dictionary should contains a '{key}' key."
            )

    if not isinstance(cfg["state_machine"], list):
        raise TypeError(
            f"State machine key in configuration should be a list of dict, each dict having two keys: "
            f"'action' and 'method'. Found '{type(cfg['state_machine'])}'."
        )

    if "initial_state" not in cfg:
        cfg["initial_state"] = param.INITIAL_STATES[0]
    else:
        if cfg["initial_state"] not in param.INITIAL_STATES:
            raise ValueError(
                f"Initial state is invalid. It should be in {param.INITIAL_STATES}."
            )

    if cfg["state_machine"]:
        for k, el in enumerate(cfg["state_machine"]):
            # Action check
            if "action" not in el:
                raise ValueError(
                    f"'action' key is missing in the {k}th element of the state machine list."
                )

            if el["action"] not in param.TRANSITIONS_METHODS:
                raise ValueError(
                    f"Element #{k} of state machine configuration: action '{el['action']}' unknown. "
                    f"It should be in {list(param.TRANSITIONS_METHODS.keys())}."
                )

            # Texture
            # Check that the parameters for rpc, image texture and utm code are given
            if el["action"] == "texture":
                if not cfg["tif_img_path"]:
                    raise ValueError(
                        "If a texturing step is asked, there should be a general configuration parameter "
                        "'tif_img_path' giving the path to the TIF image texture to process."
                    )
                if not cfg["rpc_path"]:
                    raise ValueError(
                        "If a texturing step is asked, there should be a general configuration parameter "
                        "'rpc_path' giving the path to the RPC data of the image texture."
                    )
                if not cfg["utm_code"]:
                    raise ValueError(
                        "If a texturing step is asked, there should be a general configuration parameter "
                        "'utm_code' giving the UTM code of the input point cloud or mesh for "
                        "coordinate transforming step."
                    )

            # Method check
            if "method" in el:
                # Method specified
                # Check if valid
                if el["method"] not in param.TRANSITIONS_METHODS[el["action"]]:
                    raise ValueError(
                        f"Element #{k} of state machine configuration: method '{el['method']}' unknown. "
                        f"It should be in {param.TRANSITIONS_METHODS[el['action']]}."
                    )

            else:
                # Method not specified, then select the one by default (the first one in the list)
                el["method"] = param.TRANSITIONS_METHODS[el["action"]][0]

    return cfg


def run(mesh_3d_machine: Mesh3DMachine, cfg: dict) -> Mesh:
    """
    Run the state machine

    Parameters
    ----------
    mesh_3d_machine: Mesh3DMachine
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
        logger.debug(f"Initial state: {mesh_3d_machine.initial_state}")

        # Check transitions' validity
        mesh_3d_machine.check_transitions(cfg)

        # Browse user defined steps and execute them
        for k, step in enumerate(cfg["state_machine"]):
            logger.info(
                f"Step #{k + 1}: {step['action']} with {step['method']} method"
            )
            mesh_3d_machine.run(step, cfg)

    return mesh_3d_machine.mesh_data


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

    # Write logs to disk
    logger.add(sink=os.path.join(cfg["output_dir"], "{time}_logs.txt"))
    logger.info("Configuration file checked.")

    # Read input data
    if (
        os.path.basename(cfg["input_path"]).split(".")[-1]
        in param.MESH_FILE_EXTENSIONS
    ):

        try:
            # Try reading input data as a mesh if the extension is valid
            mesh = Mesh()
            mesh.deserialize(cfg["input_path"])
            logger.debug("Input data read as a mesh format.")

        except BaseException:
            # If it does not work, try reading it with the point cloud deserializer
            # instanciate objects
            mesh = Mesh()
            mesh.pcd.deserialize(cfg["input_path"])

            logger.debug("Input data read as a point cloud format.")

    else:

        # If the extension is not a mesh extension, read the data as a point cloud and put it in a dict
        mesh = Mesh()
        mesh.pcd.deserialize(cfg["input_path"])
        logger.debug("Input data read as a point cloud format.")

    # Init state machine model
    mesh_3d_machine = Mesh3DMachine(mesh_data=mesh)

    # Run the pipeline according to the user configuration
    out_mesh = run(mesh_3d_machine, cfg)

    # Serialize data
    if out_mesh.df is not None:
        extension = "ply"
        out_filename = "processed_mesh." + extension
        out_mesh.serialize(
            filepath=os.path.join(cfg["output_dir"], out_filename),
            extension=extension,
        )
        logger.info(f"Mesh serialized as a '{extension}' file")

    else:
        extension = "las"
        out_filename = "processed_point_cloud." + extension
        out_mesh.pcd.serialize(
            filepath=os.path.join(cfg["output_dir"], out_filename),
            extension=extension,
        )
        logger.info(f"Point cloud serialized as a '{extension}' file")

    # # Save the point cloud information in a csv file
    # out_filename = "processed_point_cloud.csv"
    # out_mesh.pcd.serialize(filepath=os.path.join(cfg["output_dir"], out_filename), extension="csv")
    # logger.info(f"Point cloud serialized as a 'csv' file")

    # # Save the mesh information in a csv file
    # out_filename = "processed_mesh.csv"
    # out_mesh.serialize(filepath=os.path.join(cfg["output_dir"], out_filename), extension="csv")
    # logger.info(f"Mesh serialized as a 'csv' file")
