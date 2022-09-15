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
Console script for mesh3d.
"""

import argparse


def get_parser() -> argparse.ArgumentParser:
    """
    ArgumentParser for Mesh 3D

    Returns
    -------
    parser
    """

    # Main parser
    parser = argparse.ArgumentParser(
        description="3D textured reconstruction from remote sensing point cloud"
    )
    # parser.add_argument(
    #     "config",
    #     help="Path to a json file containing the input files paths and algorithm parameters",
    # )

    # Create subcommand parser
    subparsers = parser.add_subparsers(dest="command")

    # Reconstruction parser
    reconstruction_parser = subparsers.add_parser(
        "reconstruct",
        help="3D Reconstruction processing",
    )
    reconstruction_parser.add_argument(
        "config",
        help="Path to a json file containing the input files paths and algorithm parameters",
    )

    # Evaluation parser
    evaluation_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate mesh or point cloud",
    )
    evaluation_parser.add_argument(
        "config",
        help="Path to a json file containing the input mesh or point cloud paths to compare and the metrics to compute",
    )

    return parser


def main() -> None:
    """Console script for mesh3d."""

    # get parser
    parser = get_parser()
    args = parser.parse_args()

    # run mesh 3d pipeline
    if args.command == "reconstruct":
        # Reconstruction pipeline
        from . import mesh3d_reconstruct

        mesh3d_reconstruct.main(args.config)

    elif args.command == "evaluate":
        # Evaluation pipeline
        from . import mesh3d_evaluate

        mesh3d_evaluate.main(args.config)

    else:
        raise NotImplementedError(f"Command '{args.command}' is unknown.")


if __name__ == "__main__":
    main()
