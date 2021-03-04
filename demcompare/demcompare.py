#!/usr/bin/env python
# coding: utf8
# PYTHON_ARGCOMPLETE_OK
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
"""
demcompare aims at coregistering and comparing two dsms
"""

# Standard imports
from __future__ import print_function

import argparse
import copy

# Third party imports
import argcomplete

# DEMcompare import
import demcompare

DEFAULT_STEPS = ["coregistration", "stats", "report"]
ALL_STEPS = copy.deepcopy(DEFAULT_STEPS)


def get_parser():
    """
    ArgumentParser for demcompare
    :param None
    :return parser
    """
    parser = argparse.ArgumentParser(
        description=("Compare Digital Elevation Models")
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
        "--step",
        type=str,
        nargs="+",
        choices=ALL_STEPS,
        default=DEFAULT_STEPS,
        help="steps to choose. default: all steps",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="choose between plot show and plot save. " "default: plot save",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s {version}".format(version=demcompare.__version__),
    )
    return parser


def main():
    """
    Call demcompare's main
    """
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    demcompare.run(args.config, args.step, display=args.display)


if __name__ == "__main__":
    main()
