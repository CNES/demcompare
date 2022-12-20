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
This module contains functions to test all the
methods in the coregistration step who are supposed
to end with errors.
"""
# pylint:disable = duplicate-code

# Standard imports
import os
from tempfile import TemporaryDirectory

# Third party imports
import numpy as np
import pytest

# Demcompare imports
import demcompare
from demcompare import coregistration
from demcompare.dem_tools import SamplingSourceParameter, load_dem
from demcompare.helpers_init import mkdir_p, read_config_file, save_config_file

# Tests helpers
from tests.helpers import demcompare_test_data_path, temporary_dir


# ignore runtime for encountered value
# ignore runtime for Degrees of freedom <= 0
# ignore runtime for Mean of empty slice
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_coregistration_with_wrong_initial_disparities():
    """
    Test invalid initial disparity value
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    - invalid initial disparity
    Validation data:
    - Manually computed assertion
    Validation process:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Verify that pytest raises an error
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Load dems
    ref = load_dem(cfg["input_ref"]["path"])
    sec = load_dem(cfg["input_sec"]["path"])

    # Modify cfg's sampling_source (default is "sec")
    cfg["coregistration"]["sampling_source"] = SamplingSourceParameter.REF.value

    # Modify cfg's estimated_initial_shift
    cfg["coregistration"]["estimated_initial_shift_x"] = np.nan
    cfg["coregistration"]["estimated_initial_shift_y"] = np.nan

    coregistration_ = coregistration.Coregistration(cfg["coregistration"])

    # Compute coregistration
    with pytest.raises(ValueError):
        _ = coregistration_.compute_coregistration(sec, ref)


def test_coregistration_with_output_dir():
    """
    Test the output_dir param.
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    Validation data:
    - Manually computed assertion
    Validation process:
    - parameter output_dir being specified correctly
        - Create temporary_dir named tmp_dir
        - Loads the data present in the test root data directory
        - Creates a coregistration object and does compute_coregistration
        - Verify that coreg_sec.tif and demcompare_results.json are saved
    - parameter output_dir not being specified and save_optional_outputs
      set to True
        - Creates a new coregistration object and does compute_coregistration
        - Verify that pytest raises an error
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Load dems
    ref = load_dem(cfg["input_ref"]["path"])
    sec = load_dem(cfg["input_sec"]["path"])

    # Set output_dir correctly
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

        # test output_dir/coregistration/coreg_SEC.tif creation
        assert os.path.isfile(tmp_dir + "/coregistration/coreg_SEC.tif") is True
        # test output_dir/coregistration/coreg_SEC.tif creation
        assert os.path.isfile(tmp_dir + "/demcompare_results.json") is True

        cfg.pop("output_dir")
        # parameters save_optional_outputs
        # set to True
        cfg["coregistration"]["save_optional_outputs"] = "True"

        # Create coregistration object
        with pytest.raises(SystemExit):
            _ = coregistration.Coregistration(cfg["coregistration"])
