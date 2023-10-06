#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2023 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of cars-mesh
# (see https://github.com/CNES/cars-mesh).
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


def get_test_data_path(test_name: str) -> str:
    """
    Return full absolute path to module tests data

    :param test_name: name of test directory
    :returns: full absolute path to source tests data.
    """
    # Verify that the current path is well set
    os.chdir(os.path.dirname(__file__))

    # Get absolute path from this file (which is in "tests" directory)
    # in root_src_cars-mesh/tests/ + data/end_to_end_data

    test_data_folder = os.path.join(
        os.path.dirname(__file__), os.path.join("data")
    )
    return os.path.join(test_data_folder, test_name)


def get_module_path(directory_name: str) -> str:
    """
    Return full absolute path to package module source directory

    :param folder_name: name of directory to be located
     inside module sources
    :returns: full absolute path to module directory.
    """
    # Verify that the current path is well set
    os.chdir(os.path.dirname(__file__))
    dir_path = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))
    # Get absolute path from this file in
    # root_src/ + module directory_name
    test_data_folder = os.path.join(dir_path, "cars_mesh")

    return os.path.join(test_data_folder, directory_name)


def get_temporary_dir() -> str:
    """
    Returns path to temporary dir from TESTS_TMP_DIR environment
    variable. Defaults to /tmp
    :returns: path to tmp dir
    """
    if "TESTS_TMP_DIR" not in os.environ:
        # return default tmp dir
        return "/tmp"
    # return env defined tmp dir
    return os.environ["TESTS_TMP_DIR"]
