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
Main Reconstruct pipeline module of cars-mesh tool.
"""

import logging
import os

from . import setup_logging
from .config import check_config, save_config_file
from .state_machine import CarsMeshMachine
from .tools.handlers import Mesh, read_input_path


def run(cars_mesh_machine: CarsMeshMachine, cfg: dict) -> Mesh:
    """
    Run the state machine

    Parameters
    ----------
    cars_mesh_machine: CarsMeshMachine
        CARS-MESH state machine model
    cfg: dict
        Configuration dictionary

    Returns
    -------
    mesh: Mesh
        Mesh object
    """

    if not cfg["state_machine"]:
        # There is no step given to the state machine
        logging.warning("State machine is empty. Returning the initial data.")

    else:
        logging.debug(f"Initial state: {cars_mesh_machine.initial_state}")

        # Check transitions' validity
        cars_mesh_machine.check_transitions(cfg)

        # Browse user defined steps and execute them
        for k, step in enumerate(cfg["state_machine"]):
            # Logger
            logging.info(
                f"Step #{k + 1}: {step['action']} with {step['method']} method"
            )

            # Run action
            cars_mesh_machine.run(step, cfg)

            # (Optional) Save intermediate results to disk if asked
            if "save_output" in step:
                if step["save_output"]:
                    # Create directory to save intermediate results
                    intermediate_folder = os.path.join(
                        cfg["output_dir"], "intermediate_results"
                    )
                    os.makedirs(intermediate_folder, exist_ok=True)
                    if k == 0 and os.listdir(intermediate_folder):
                        logging.warning(
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
                    if cars_mesh_machine.mesh_data.df is not None:
                        # Mesh serialisation
                        extension = "ply"
                        cars_mesh_machine.mesh_data.serialize(
                            filepath=intermediate_filepath + "." + extension,
                            extension=extension,
                        )
                    else:
                        # Point cloud serialisation
                        extension = "laz"
                        cars_mesh_machine.mesh_data.pcd.serialize(
                            filepath=intermediate_filepath + "." + extension,
                            extension=extension,
                        )

                    # Logger debug
                    logging.debug(
                        f"Step #{k + 1}: Results saved in "
                        f"'{intermediate_filepath  + '.' + extension}'."
                    )

    return cars_mesh_machine.mesh_data


def main(cfg_path: str) -> None:
    """
    Main function to apply cars-mesh pipeline

    Parameters
    ----------
    cfg_path: str
        Path to the JSON configuration file
    """
    # To avoid having a logger INFO for each state machine step
    logging.getLogger("transitions").setLevel(logging.WARNING)

    # Check the validity of the config and update cfg with absolute paths
    cfg = check_config(cfg_path)

    # Create output directory and update config
    output_dir = os.path.abspath(cfg["output_dir"])
    cfg["output_dir"] = output_dir

    # Create output_dir
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Save the configuration in the output dir
    # with inputs and output absolute paths
    save_config_file(
        os.path.join(cfg["output_dir"], os.path.basename(cfg_path)), cfg
    )

    # Write logs to disk
    setup_logging.add_log_file(cfg["output_dir"], "cars_mesh")
    logging.info("Configuration file checked.")

    # Read input data
    mesh = read_input_path(cfg["input_path"])

    # Init state machine model
    cars_mesh_machine = CarsMeshMachine(
        mesh_data=mesh, initial_state=cfg["initial_state"]
    )

    # Run the pipeline according to the user configuration
    out_mesh = run(cars_mesh_machine, cfg)

    # Serialize data
    # Check if user specified an output name
    # otherwise assign a default one
    if "output_name" not in cfg:
        cfg["output_name"] = "output_cars-mesh"  # default output name
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
        logging.info(f"Mesh serialized as a '{extension}' file")
    else:
        # Point Cloud Serialisation
        out_mesh.pcd.serialize(
            filepath=os.path.join(cfg["output_dir"], out_filename),
            extension=extension,
        )
        logging.info(f"Point cloud serialized as a '{extension}' file")
