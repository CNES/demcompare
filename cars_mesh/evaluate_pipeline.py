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
Main pipeline module for evaluation.
"""

# Standard imports
import json
import logging
import os
import shutil

# Third party imports
import pandas as pd
from tqdm import tqdm

# Cars-mesh imports
from . import param, setup_logging
from .tools.handlers import Mesh, read_input_path
from .tools.metrics import PointCloudMetrics


def check_config(cfg_path: str) -> dict:
    """
    Check if the evaluate config is valid and readable

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
            f"Configuration path is invalid. It should be a string but "
            f"got '{type(cfg_path)}'."
        )

    if os.path.basename(cfg_path).split(".")[-1] != "json":
        raise ValueError(
            f"Configuration path should be a JSON file with extension "
            f"'.json'. Found '{os.path.basename(cfg_path).split('.')[-1]}'."
        )

    # Read JSON file
    with open(cfg_path, "r", encoding="utf-8") as cfg_file:
        cfg = json.load(cfg_file)

    # Check the validity of the content
    for input_path in ["input_path_1", "input_path_2"]:
        if input_path not in cfg:
            raise ValueError(
                f"Configuration dictionary is missing the '{input_path}' "
                f"field."
            )

        if not isinstance(cfg[input_path], str):
            raise TypeError(
                f"'{input_path}' is invalid. It should be a string but "
                f"got '{type(cfg[input_path])}'."
            )

        if os.path.basename(cfg[input_path]).split(".")[-1] not in (
            param.PCD_FILE_EXTENSIONS + param.MESH_FILE_EXTENSIONS
        ):
            raise ValueError(
                f"'{input_path}' extension is invalid. "
                f"It should be in "
                f"{param.PCD_FILE_EXTENSIONS + param.MESH_FILE_EXTENSIONS}"
            )

    if "output_dir" not in cfg:
        raise ValueError(
            "Configuration dictionary should contains a 'output_dir' key."
        )

    return cfg


def run(cfg: dict, mesh_data_1: Mesh, mesh_data_2: Mesh) -> pd.DataFrame:
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

    for mode in tqdm(metrics.modes, position=0, leave=False, desc="Mode"):
        for metric_name, metric_function in tqdm(
            metrics.metrics.items(), position=1, leave=False, desc="Metrics"
        ):
            res = metric_function(mode)

            if isinstance(res, float):
                data.append((metric_name, mode, res, None))
            else:
                data.append((metric_name, mode, *list(res)))

    df_metrics = pd.DataFrame.from_records(data, columns=columns)

    # Serialize visu
    metrics.visualize_distances(cfg["output_dir"])
    logging.info(
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

    # Copy the configuration file in the output dir
    os.makedirs(cfg["output_dir"], exist_ok=True)
    shutil.copy(
        cfg_path,
        os.path.join(
            os.path.join(cfg["output_dir"], os.path.basename(cfg_path))
        ),
    )

    # Write logs to disk
    setup_logging.add_log_file(cfg["output_dir"], "evaluate_logs")
    logging.info("Configuration file checked.")

    # Read input data
    mesh_data_1 = read_input_path(cfg["input_path_1"])
    mesh_data_2 = read_input_path(cfg["input_path_2"])

    # Run the pipeline according to the user configuration
    df_metrics = run(cfg, mesh_data_1, mesh_data_2)

    # Serialize metrics
    filename = os.path.join(cfg["output_dir"], "metrics.csv")
    df_metrics.to_csv(filename, index=False)
    logging.info(f"Metrics serialized in '{filename}'")
