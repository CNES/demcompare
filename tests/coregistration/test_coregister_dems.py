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
# Disable the protected-access to test the functions

"""
This module contains functions to test
the coregistration._coregister_dems_algorithm function
"""
# pylint:disable=protected-access
# pylint:disable=duplicate-code
# Standard imports
import os

# Third party imports
import numpy as np
import pytest

# Demcompare imports
from demcompare import coregistration, dem_tools
from demcompare.helpers_init import read_config_file

# Tests helpers
from tests.helpers import demcompare_test_data_path


@pytest.mark.unit_tests
def test_coregister_dems_algorithm_gironde_sampling_sec():
    """
    Test the _coregister_dems_algorithm function of
    the Nuth et Kaab class.
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    - sampling value : "sec"
    Validation data:
     - Manually computed ground truth : x_offset, y_offset z_offset and rotation
    Validation process:
    - Loads the data present in the test root data directory
    - Reprojects DEMs with sampling_source "sec"
    - Creates a coregistration object and does _coregister_dems_algorithm
    - Checked parameters are:
        - x_offset
        - y_offset
        - z_offset
        - rotation
    """
    # Test with "gironde_test_data" test root
    # input DEMs and sampling value sec -----------------------------

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Load dems
    ref = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec = dem_tools.load_dem(cfg["input_sec"]["path"])
    sampling_source = "sec"

    # Define ground truth outputs
    rotation = None
    x_offset = -1.4366
    y_offset = -0.4190
    z_offset = -3.4025

    # Reproject and crop DEMs
    (
        reproj_crop_dem,
        reproj_crop_ref,
        _,
    ) = dem_tools.reproject_dems(sec, ref, sampling_source=sampling_source)

    # Coregistration configuration is the following :
    # "coregistration": {
    #    "method_name": "nuth_kaab_internal",
    #    "number_of_iterations": 6,
    #    "estimated_initial_shift_x": 0,
    #    "estimated_initial_shift_y": 0
    # }
    # Create coregistration object
    coregistration_ = coregistration.Coregistration(cfg["coregistration"])
    # Run _coregister_dems_algorithm
    transform, _, _ = coregistration_._coregister_dems_algorithm(
        reproj_crop_dem, reproj_crop_ref
    )

    # Test that the outputs match the ground truth
    assert rotation == transform.rotation
    np.testing.assert_allclose(x_offset, transform.x_offset, rtol=1e-02)
    np.testing.assert_allclose(y_offset, transform.y_offset, rtol=1e-02)
    np.testing.assert_allclose(z_offset, transform.z_offset, rtol=1e-02)


@pytest.mark.unit_tests
def test_bounds_in_coregister_dems_algorithm_gironde_sampling_sec():
    """
    Test the _coregister_dems_algorithm function of
    the Nuth et Kaab class.
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    - sampling value : "sec"
    Validation data:
     - Manually computed ground truth bounds: ulx, uly, lrx, lry
    Validation process:
    - Loads the data present in the test root data directory
    - Reprojects DEMs with sampling_source "sec"
    - Creates a coregistration object and does _coregister_dems_algorithm
    -  Checked parameters are:
        - x_offset
        - y_offset
        - z_offset
        - rotation
    """
    # Test with "gironde_test_data" test root
    # input DEMs and sampling value sec -----------------------------

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Load dems
    ref = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec = dem_tools.load_dem(cfg["input_sec"]["path"])
    sampling_source = "sec"

    # Define ground truth outputs
    ulx = 599536.6809346128
    uly = 5090035.483811519
    lrx = 688536.6809346128
    lry = 4990535.483811519

    # Reproject and crop DEMs
    (
        reproj_crop_dem,
        reproj_crop_ref,
        _,
    ) = dem_tools.reproject_dems(sec, ref, sampling_source=sampling_source)

    # Coregistration configuration is the following :
    # "coregistration": {
    #    "method_name": "nuth_kaab_internal",
    #    "number_of_iterations": 6,
    #    "estimated_initial_shift_x": 0,
    #    "estimated_initial_shift_y": 0
    # }
    # Create coregistration object
    coregistration_ = coregistration.Coregistration(cfg["coregistration"])
    # Run _coregister_dems_algorithm
    (
        _,
        coreg_sec_dataset,
        coreg_ref_dataset,
    ) = coregistration_._coregister_dems_algorithm(
        reproj_crop_dem, reproj_crop_ref
    )

    # Test that the outputs match the ground truth
    np.testing.assert_allclose(
        coreg_sec_dataset.bounds, (ulx, uly, lrx, lry), rtol=1e-02
    )
    np.testing.assert_allclose(
        coreg_ref_dataset.bounds, (ulx, uly, lrx, lry), rtol=1e-02
    )


@pytest.mark.unit_tests
def test_coregister_dems_algorithm_gironde_sampling_ref():
    """
    Test the _coregister_dems_algorithm function of
    the Nuth et Kaab class.
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    - sampling value : "ref"
    Validation data:
     - Manually computed ground truth : x_offset, y_offset z_offset and rotation
    Validation process:
    - Loads the data present in the test root data directory
    - Reprojects DEMs with sampling_source "ref"
    - Creates a coregistration object and does _coregister_dems_algorithm
    - Checked parameters are:
        - x_offset
        - y_offset
        - z_offset
        - rotation
    """

    # Test with "gironde_test_data" test root
    # input DEMs and sampling value ref -----------------------------

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Load dems
    ref = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec = dem_tools.load_dem(cfg["input_sec"]["path"])
    sampling_source = "ref"

    # Define ground truth outputs
    rotation = None
    x_offset = -5.44706
    y_offset = -1.18139
    z_offset = -0.53005

    # Reproject and crop DEMs
    (
        reproj_crop_dem,
        reproj_crop_ref,
        _,
    ) = dem_tools.reproject_dems(sec, ref, sampling_source=sampling_source)

    # Coregistration configuration is the following :
    # "coregistration": {
    #    "method_name": "nuth_kaab_internal",
    #    "number_of_iterations": 6,
    #    "estimated_initial_shift_x": 0,
    #    "estimated_initial_shift_y": 0
    # }
    # Create coregistration object
    coregistration_ = coregistration.Coregistration(cfg["coregistration"])
    # Run coregister_dems
    (transform, _, _,) = coregistration_._coregister_dems_algorithm(
        reproj_crop_dem, reproj_crop_ref
    )

    # Test that the outputs match the ground truth
    assert rotation == transform.rotation
    np.testing.assert_allclose(x_offset, transform.x_offset, rtol=1e-02)
    np.testing.assert_allclose(y_offset, transform.y_offset, rtol=1e-02)
    np.testing.assert_allclose(z_offset, transform.z_offset, rtol=1e-02)
