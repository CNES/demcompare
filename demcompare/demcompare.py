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
# PYTHON_ARGCOMPLETE_OK
"""
demcompare aims at coregistering and comparing two dsms
"""

# Standard imports
from __future__ import print_function

import argparse

# Third party imports
import argcomplete

# DEMcompare import
import demcompare


def get_parser():
    """
    ArgumentParser for demcompare

    :return: parser
    """
    parser = argparse.ArgumentParser(
        description=("Compare Digital Elevation Models"),
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "config",
        metavar="config.json",
        help=(
            "path to a json file containing the paths to "
            "input and output files and the algorithm "
            "parameters"
        ),
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: INFO. Should be one of "
        "(DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser


def main():
    """
    Call demcompare's main
    """
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    demcompare.run(args.config, loglevel=args.loglevel)


if __name__ == "__main__":
    main()
