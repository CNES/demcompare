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
Console script for cars-mesh-evaluate.
"""

# Standard imports
import argparse
import logging
import sys
import traceback

# Cars-mesh imports
from cars_mesh import __version__, evaluate_pipeline, setup_logging


def get_parser() -> argparse.ArgumentParser:
    """
    ArgumentParser for CARS-MESH

    Returns
    parser
    """

    # Evaluate parser
    parser = argparse.ArgumentParser(description="Evaluate mesh or point cloud")

    parser.add_argument(
        "config",
        help="Path to a json file containing the input mesh or point cloud "
        "paths to compare and the metrics to compute",
    )

    parser.add_argument(
        "--loglevel",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="Logger level cars-mesh evaluate (default: INFO. Should be one of "
        "(DEBUG, INFO, INFO, WARNING, ERROR, CRITICAL)",
    )

    # General arguments at first level
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def main() -> None:
    """Console script for cars-mesh-evaluate."""

    # get parser of cars-mesh-evaluate
    parser = get_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])

    # set logging for cars-mesh-evaluate
    setup_logging.setup_logging(default_level=args.loglevel)
    logging.debug("Show argparse arguments: %s", args)

    try:
        # use a global try/except to cath
        # run cars-mesh main evaluate pipeline
        evaluate_pipeline.main(args.config)
    except Exception:  # pylint: disable=broad-except
        logging.error(" cars-mesh-evaluate %s", traceback.format_exc())


if __name__ == "__main__":
    main()
