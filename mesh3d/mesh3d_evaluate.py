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
Main module for evaluation.
"""

import json
import os

import pandas
import pandas as pd
from loguru import logger

from . import param
from .tools.handlers import Mesh
from .tools.metrics import PointCloudMetrics


def read_input_path(input_path: str) -> Mesh:
    """
    Read input path as either a PointCloud or a Mesh object

    Parameters
    ----------
    input_path: str
        Input path to read

    Returns
    -------
    mesh: Mesh
        Mesh object
    """
    if (
        os.path.basename(input_path).split(".")[-1]
        in param.MESH_FILE_EXTENSIONS
    ):

        try:
            # Try reading input data as a mesh if the extension is valid
            mesh = Mesh()
            mesh.deserialize(input_path)
            logger.debug("Input data read as a mesh format.")

        except BaseException:
            # If it does not work, try reading it with the point cloud deserializer
            # instanciate objects
            mesh = Mesh()
            mesh.pcd.deserialize(input_path)

            logger.debug("Input data read as a point cloud format.")

    else:

        # If the extension is not a mesh extension, read the data as a point cloud and put it in a dict
        mesh = Mesh()
        mesh.pcd.deserialize(input_path)
        logger.debug("Input data read as a point cloud format.")

    return mesh


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
    for input_path in ["input_path_1", "input_path_2"]:

        if input_path not in cfg:
            raise ValueError(
                f"Configuration dictionary is missing the '{input_path}' field."
            )

        else:
            if not isinstance(cfg[input_path], str):
                raise TypeError(
                    f"'{input_path}' is invalid. It should be a string but got '{type(cfg[input_path])}'."
                )

            if os.path.basename(cfg[input_path]).split(".")[-1] not in (
                param.PCD_FILE_EXTENSIONS + param.MESH_FILE_EXTENSIONS
            ):
                raise ValueError(
                    f"'{input_path}' extension is invalid. "
                    f"It should be in {param.PCD_FILE_EXTENSIONS + param.MESH_FILE_EXTENSIONS}."
                )

    if "output_dir" not in cfg:
        raise ValueError(
            f"Configuration dictionary should contains a 'output_dir' key."
        )

    return cfg


def run(cfg: dict, mesh_data_1: Mesh, mesh_data_2: Mesh) -> pandas.DataFrame:
    """
    Run evaluation

    Parameters
    ----------
    cfg: dict
        Configuration dictionary
    mesh_data_1: Mesh
    mesh_data_2: Mesh

    Returns
    -------
    df_metrics: pandas DataFrame
        Metrics
    """
    # Init metrics
    metrics = PointCloudMetrics(mesh_data_1.pcd, mesh_data_2.pcd)

    # Compute them all and push them in a pandas DataFrame for easy saving
    # Create a dictionary that will be converted to a pandas DataFrame
    # Columns
    columns = ["metric_name", "mode", "1vs2_or_symmetric", "2vs1"]
    data = []

    for mode in metrics.modes:
        for metric_name in metrics.metrics:
            res = metrics.metrics[metric_name](mode)

            if isinstance(res, float):
                data.append((metric_name, mode, res, None))
            else:
                data.append((metric_name, mode, *list(res)))

    df_metrics = pd.DataFrame.from_records(data, columns=columns)

    # Serialize visu
    metrics.visualize_distances(cfg["output_dir"])
    logger.info(
        f"Point cloud distances were written to disk in '{cfg['output_dir']}'."
    )

    return df_metrics


def main(cfg_path: str) -> None:
    """
    Main function to evaluate a point cloud or a mesh

    Parameters
    ----------
    cfg_path: str
        Path to the JSON configuration file
    """
    # Check the validity of the config path
    cfg = check_config(cfg_path)

    # Write logs to disk
    logger.add(sink=os.path.join(cfg["output_dir"], "{time}_logs.txt"))
    logger.info("Configuration file checked.")

    # Read input data
    mesh_data_1 = read_input_path(cfg["input_path_1"])
    mesh_data_2 = read_input_path(cfg["input_path_2"])

    # Run the pipeline according to the user configuration
    df_metrics = run(cfg, mesh_data_1, mesh_data_2)

    # Serialize metrics
    filename = os.path.join(cfg["output_dir"], "metrics.csv")
    df_metrics.to_csv(filename, index=False)
    logger.info(f"Metrics serialized in '{filename}'")
