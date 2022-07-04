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
Console script for mesh_3d.
"""

import argparse
from . import mesh_3d


def get_parser() -> argparse.ArgumentParser:
    """
    ArgumentParser for Mesh 3D

    Returns
    -------
    parser
    """
    parser = argparse.ArgumentParser(description="3D textured reconstruction from remote sensing point cloud")
    parser.add_argument(
        "config",
        help="Path to a json file containing the input files paths and algorithm parameters",
    )
    # parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    return parser


def main():
    """Console script for mesh_3d."""

    # get parser
    parser = get_parser()
    args = parser.parse_args()

    # run mesh 3d pipeline
    mesh_3d.main(args.config)

    # # DEBUG
    # cfg_path = "/home/chthen/Documents/CNES_RT_Surfaces_3D/code/configs/baseline_pipeline_config.json"
    # mesh_3d.main(cfg_path)


if __name__ == "__main__":
    main()
