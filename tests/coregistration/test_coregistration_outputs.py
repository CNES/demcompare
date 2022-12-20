#!/usr/bin/env python
# coding: utf8
# Disable the protected-access to test the functions
# pylint:disable=protected-access
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
This module contains functions to test the ouputs
of the coregistration method.
"""

import glob

# Standard imports
import os
from tempfile import TemporaryDirectory

# Demcompare imports
import demcompare
from demcompare import coregistration
from demcompare.dem_tools import load_dem
from demcompare.helpers_init import mkdir_p, read_config_file, save_config_file

# Tests helpers
from tests.helpers import demcompare_test_data_path, temporary_dir


def test_coregistration_save_optional_reprojection():
    """
    Test the coregistration save_optional_outputs parameter
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    Validation data:
    - Manually computed gt_truth_list_files
    Validation process:
    - parameter save_optional_outputs set to True in config
    - Create temporary_dir named tmp_dir
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Verify that all files in gt_truth_list_files are saves in tmp_dir
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Manually set the saving of internal dems to True
    cfg["coregistration"]["save_optional_outputs"] = "True"
    # remove useless statistics part
    cfg.pop("statistics")

    gt_truth_list_files = [
        "reproj_coreg_SEC.tif",
        "reproj_coreg_REF.tif",
        "reproj_SEC.tif",
        "reproj_REF.tif",
    ]

    # Load dems
    ref = load_dem(cfg["input_ref"]["path"])
    sec = load_dem(cfg["input_sec"]["path"])

    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        mkdir_p(tmp_dir)
        # Modify test's output dir in configuration to tmp test dir
        cfg["output_dir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, cfg)

        # Run demcompare with "srtm_test_data"
        # Put output_dir in coregistration dict config
        demcompare.run(tmp_cfg_file)
        tmp_cfg = read_config_file(tmp_cfg_file)
        # Create Coregistration object
        coregistration_ = coregistration.Coregistration(
            tmp_cfg["coregistration"]
        )

        # compute coregistration
        _ = coregistration_.compute_coregistration(sec, ref)

        # test output_dir/coregistration creation
        assert os.path.exists(tmp_dir + "/coregistration") is True

        # get all files saved en output_dir/coregistration
        list_test = [
            os.path.basename(x)
            for x in glob.glob(tmp_dir + "/coregistration/*")
        ]
        # test all files in gt_truth_list_files are in coregistration directory
        assert all(file in list_test for file in gt_truth_list_files) is True


def test_coregistration_save_optional_outputs():
    """
    Test the coregistration save_optional_outputs parameter
    for the iteration plots of Nuth et kaab.
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    Validation data:
    - Manually computed gt_truth_list_files
    Validation process:
    - parameter save_optional_outputs set to True in config
        - Create temporary_dir named tmp_dir
        - Loads the data present in the test root data directory
        - Creates a coregistration object and does compute_coregistration
        - Verify that all files in gt_truth_list_files are saves in tmp_dir
    - parameter save_optional_outputs set to False in config
        - Create temporary_dir named tmp_dir
        - Loads the data present in the test root data directory
        - Creates a coregistration object and does compute_coregistration
        - Verify that the iteration plots of Nuth et kaab aren't saved
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # remove useless statistics part
    cfg.pop("statistics")
    # Set save_optional_outputs to True
    cfg["coregistration"]["save_optional_outputs"] = "True"

    gt_truth_list_files = [
        "ElevationDiff_AfterCoreg.png",
        "ElevationDiff_BeforeCoreg.png",
        "nuth_kaab_iter#0.png",
        "nuth_kaab_iter#1.png",
        "nuth_kaab_iter#2.png",
        "nuth_kaab_iter#3.png",
        "nuth_kaab_iter#4.png",
        "nuth_kaab_iter#5.png",
    ]

    # Load dems
    ref = load_dem(cfg["input_ref"]["path"])
    sec = load_dem(cfg["input_sec"]["path"])

    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        mkdir_p(tmp_dir)
        # Modify test's output dir in configuration to tmp test dir
        cfg["output_dir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, cfg)

        # Run demcompare with "srtm_test_data"
        # Put output_dir in coregistration dict config
        demcompare.run(tmp_cfg_file)
        tmp_cfg = read_config_file(tmp_cfg_file)
        # Create Coregistration object
        coregistration_ = coregistration.Coregistration(
            tmp_cfg["coregistration"]
        )

        # compute coregistration
        _ = coregistration_.compute_coregistration(sec, ref)

        # test output_dir/coregistration/nuth_kaab_tmp_dir/ creation
        assert (
            os.path.exists(tmp_dir + "/coregistration/nuth_kaab_tmp_dir/")
            is True
        )

        # get all files saved in output_dir/coregistration/nuth_kaab_tmp_dir/
        list_test = [
            os.path.basename(x)
            for x in glob.glob(tmp_dir + "/coregistration/nuth_kaab_tmp_dir/*")
        ]
        # test all files in gt_truth_list_files are in coregistration directory
        assert all(file in list_test for file in gt_truth_list_files) is True

    # Test with save_optional_outputs set to False
    cfg["coregistration"]["save_optional_outputs"] = "False"

    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        mkdir_p(tmp_dir)
        # Modify test's output dir in configuration to tmp test dir
        cfg["output_dir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, cfg)

        # Run demcompare with "srtm_test_data"
        # Put output_dir in coregistration dict config
        demcompare.run(tmp_cfg_file)
        tmp_cfg = read_config_file(tmp_cfg_file)
        # Create Coregistration object
        coregistration_ = coregistration.Coregistration(
            tmp_cfg["coregistration"]
        )

        # compute coregistration
        _ = coregistration_.compute_coregistration(sec, ref)

        # test output_dir/coregistration/nuth_kaab_tmp_dir/ creation
        assert (
            os.path.exists(tmp_dir + "/coregistration/nuth_kaab_tmp_dir/")
            is True
        )

        # get all files saved in output_dir/coregistration/nuth_kaab_tmp_dir/
        list_test = [
            os.path.basename(x)
            for x in glob.glob(tmp_dir + "/coregistration/nuth_kaab_tmp_dir/*")
        ]
        # test list_test is empty
        assert list_test == []
