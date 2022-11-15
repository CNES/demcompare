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
"""
Helpers shared testing generic module:
contains global shared generic functions for tests/*.py
"""

# Standard imports
import os
from typing import List

# Third party imports
import numpy as np
import rasterio as rio

# Define tests tolerance
TEST_TOL = 1e-02


def demcompare_test_data_path(test_name: str) -> str:
    """
    Return full absolute path to demcompare's tests data

    :param test_name: name of test directory
    :returns: full absolute path to demcompare test data.
    """
    # Verify that the current path is well set
    os.chdir(os.path.dirname(__file__))

    # Get absolute path from this file
    # in root_src_demcompare/tests/ + data/end_to_end_data

    test_data_folder = os.path.join(
        os.path.dirname(__file__), os.path.join("data", "end_to_end_data")
    )
    return os.path.join(test_data_folder, test_name)


def demcompare_path(directory_name: str) -> str:
    """
    Return full absolute path to demcompare's desired directory

    :param folder_name: name of directory to be located
     inside demcompare/demcompare
    :returns: full absolute path to demcompare directory.
    """
    # Verify that the current path is well set
    os.chdir(os.path.dirname(__file__))
    dir_path = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))
    # Get absolute path from this file in
    # root_src_demcompare/demcompare/ + directory_name
    test_data_folder = os.path.join(dir_path, "demcompare")

    return os.path.join(test_data_folder, directory_name)


def notebooks_demcompare_path(notebook_name: str) -> str:
    """
    Return full absolute path to demcompare's desired directory

    :param notebook_name: name of notebook to be located
     inside demcompare/notebooks
    :returns: full absolute path to notebooks directory.
    """
    # Verify that the current path is well set
    os.chdir(os.path.dirname(__file__))
    dir_path = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))
    # Get absolute path from this file in
    # root_src_demcompare/demcompare/ + notebook_name
    notebooks_folder = os.path.join(dir_path, "notebooks")

    return os.path.join(notebooks_folder, notebook_name)


def read_csv_file(csv_file: str) -> List[np.ndarray]:
    """
    Read a csv file and save its number values to float

    :param csv_file: path to a csv file
    :type csv_file: string
    :returns: List of floats of input csv file
    :rtype: List[np.ndarray]
    """
    output_file = []

    with open(csv_file, "r", encoding="utf-8") as file_handle:
        lines = file_handle.readlines()

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


def assert_same_images(
    actual: str, expected: str, rtol: float = 0, atol: float = 0
):
    """
    Compare two image files with assertion:
    * same height, width, transform, crs
    * assert_allclose() on numpy buffers

    :param actual: image to compare
    :param expected: reference image to compare
    :param rtol: relative tolerance
    :param atol: absolute tolerance
    """
    with rio.open(actual) as rio_actual:
        with rio.open(expected) as rio_expected:
            np.testing.assert_equal(rio_actual.width, rio_expected.width)
            np.testing.assert_equal(rio_actual.height, rio_expected.height)
            np.testing.assert_allclose(
                np.array(rio_actual.transform),
                np.array(rio_expected.transform),
                atol=atol,
            )
            assert rio_actual.crs == rio_expected.crs
            assert rio_actual.nodata == rio_expected.nodata
            np.testing.assert_allclose(
                rio_actual.read(), rio_expected.read(), rtol=rtol, atol=atol
            )


def temporary_dir() -> str:
    """
    Returns path to temporary dir from DEMCOMPARE_TMP_DIR environment
    variable. Defaults to /tmp
    :returns: path to tmp dir
    """
    if "DEMCOMPARE_TMP_DIR" not in os.environ:
        # return default tmp dir
        return "/tmp"
    # return env defined tmp dir
    return os.environ["DEMCOMPARE_TMP_DIR"]
