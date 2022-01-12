#!/usr/bin/env python
# pylint:disable=unspecified-encoding
# pylint:disable=consider-using-with
# coding: utf8
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
Helpers shared testing generic module:
contains global shared generic functions for tests/*.py
"""

import json

# Standard imports
import os
from typing import Dict

# Third party imports
import numpy as np
import rasterio as rio


def demcompare_test_data_path(test_name: str):
    """
    Return full absolute path to demcompare's tests data
    """
    half_path = os.path.join("tests/tests_data", test_name)
    return os.path.join(os.getcwd(), half_path)


def read_config_file(config_file: str) -> Dict[str, dict]:
    """
    Read a json configuration file

    :param config_file: path to a json file
    :type config_file: string
    :return user_cfg: configuration dictionary
    :rtype: dict
    """
    with open(config_file, "r") as file_:
        user_cfg = json.load(file_)
    return user_cfg


def read_csv_file(csv_file: str):
    """
    Read a csv file and save its number values to float

    :param csv_file: path to a csv file
    :type csv_file: string
    """
    lines = open(csv_file, "r").readlines()
    output_file = []

    for idx, line in enumerate(lines):
        # Obtain colums
        cols = line.split(",")
        # Last column ends with \n
        cols[-1] = cols[-1].split("\n")[0]
        # First line are titles
        if idx == 0:
            continue
        # If it is the stats csv, do not convert to float first col
        if len(cols) > 2:
            output_file.append(np.array(cols[1:], dtype=float))
            continue
        # Convert to float
        output_file.append(np.array(cols, dtype=float))
    return output_file


def assert_same_images(actual, expected, rtol=0, atol=0):
    """
    Compare two image files with assertion:
    * same height, width, transform, crs
    * assert_allclose() on numpy buffers
    """
    with rio.open(actual) as rio_actual:
        with rio.open(expected) as rio_expected:
            np.testing.assert_equal(rio_actual.width, rio_expected.width)
            np.testing.assert_equal(rio_actual.height, rio_expected.height)
            assert rio_actual.transform == rio_expected.transform
            assert rio_actual.crs == rio_expected.crs
            assert rio_actual.nodata == rio_expected.nodata
            np.testing.assert_allclose(
                rio_actual.read(), rio_expected.read(), rtol=rtol, atol=atol
            )
