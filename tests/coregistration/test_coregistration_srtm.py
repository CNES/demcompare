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
methods in the coregistration step step with srtm datas.
"""
# pylint:disable = duplicate-code
# Standard imports
import os
from typing import Dict

# Third party imports
import numpy as np

# Demcompare imports
from demcompare import coregistration
from demcompare.dem_tools import SamplingSourceParameter, load_dem
from demcompare.helpers_init import read_config_file

# Tests helpers
from tests.helpers import demcompare_test_data_path


def test_compute_coregistration_with_srtm_sampling_sec_and_initial_disparity():
    """
    Test the compute_coregistration function
    Input data:
    - input DEMs present in "srtm_test_data" test root data directory
    - sampling value : "sec"
    - coregistration method : "Nuth et kaab"
    - non-zero initial disparity
    Validation data:
    - Manually computed gt_xoff, gt_yoff,
      gt_sampling_source and gt_plani_results
    Validation process:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Checked parameters on the demcompare_results output dict are:
        - output Transform correct offsets
        - considered sampling source "sec"
        - offsets gt_xoff, gt_yoff
        - bias
        - gdal_translate_bounds
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("srtm_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Modify cfg's estimated_initial_shift
    cfg["coregistration"]["estimated_initial_shift_x"] = 2
    cfg["coregistration"]["estimated_initial_shift_y"] = 3

    # Load dems
    ref = load_dem(cfg["input_ref"]["path"])
    sec = load_dem(cfg["input_sec"]["path"])

    # Define ground truth offsets
    gt_xoff = 0.99999
    gt_yoff = 2.00000
    gt_sampling_source = "sec"
    gt_plani_results: Dict = {
        "dx": {
            "nuth_offset": 1.0,
            "total_offset": 3.0,
            "unit_offset": "px",
            "total_bias_value": 0.0025,
            "unit_bias_value": "deg",
        },
        "dy": {
            "nuth_offset": 2.0,
            "total_offset": 5.0,
            "unit_offset": "px",
            "total_bias_value": 0.00417,
            "unit_bias_value": "deg",
        },
        "gdal_translate_bounds": {
            "ulx": 40.00083,
            "uly": 39.99833,
            "lrx": 40.83167,
            "lry": 39.16917,
        },
    }

    coregistration_ = coregistration.Coregistration(cfg["coregistration"])

    # Compute coregistration
    transform = coregistration_.compute_coregistration(sec, ref)
    # Get coregistration results
    demcompare_results = coregistration_.demcompare_results

    # Test that the output offsets and bias are the same as gt
    np.testing.assert_allclose(gt_xoff, transform.x_offset, rtol=1e-02)
    np.testing.assert_allclose(gt_yoff, transform.y_offset, rtol=1e-02)
    assert gt_sampling_source == coregistration_.sampling_source
    np.testing.assert_allclose(
        gt_plani_results["dx"]["nuth_offset"],
        demcompare_results["coregistration_results"]["dx"]["nuth_offset"],
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_plani_results["dx"]["total_bias_value"],
        demcompare_results["coregistration_results"]["dx"]["total_bias_value"],
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_plani_results["dy"]["nuth_offset"],
        demcompare_results["coregistration_results"]["dy"]["nuth_offset"],
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_plani_results["dy"]["total_bias_value"],
        demcompare_results["coregistration_results"]["dy"]["total_bias_value"],
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_plani_results["gdal_translate_bounds"]["ulx"],
        demcompare_results["coregistration_results"]["gdal_translate_bounds"][
            "ulx"
        ],
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        gt_plani_results["gdal_translate_bounds"]["uly"],
        demcompare_results["coregistration_results"]["gdal_translate_bounds"][
            "uly"
        ],
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        gt_plani_results["gdal_translate_bounds"]["lrx"],
        demcompare_results["coregistration_results"]["gdal_translate_bounds"][
            "lrx"
        ],
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        gt_plani_results["gdal_translate_bounds"]["lry"],
        demcompare_results["coregistration_results"]["gdal_translate_bounds"][
            "lry"
        ],
        rtol=1e-03,
    )


def test_compute_coregistration_with_srtm_sampling_ref_and_initial_disparity():
    """
    Test the compute_coregistration function
    Input data:
    - input DEMs present in "srtm_test_data" test root data directory
    - sampling value : "ref"
    - coregistration method : "Nuth et kaab"
    - non-zero initial disparity
    Validation data:
    - Manually computed gt_xoff, gt_yoff,
      gt_sampling_source and gt_plani_results
    Validation process:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Checked parameters on the demcompare_results output dict are:
        - output Transform correct offsets
        - considered sampling source "ref"
        - offsets gt_xoff, gt_yoff
        - bias
        - gdal_translate_bounds
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("srtm_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Modify cfg's sampling_source (default is "sec")
    cfg["coregistration"]["sampling_source"] = SamplingSourceParameter.REF.value

    # Modify cfg's estimated_initial_shift
    cfg["coregistration"]["estimated_initial_shift_x"] = 2
    cfg["coregistration"]["estimated_initial_shift_y"] = 3

    # Load dems
    ref = load_dem(cfg["input_ref"]["path"])
    sec = load_dem(cfg["input_sec"]["path"])

    # Define ground truth offsets
    gt_xoff = 0.99699
    gt_yoff = 1.99000
    gt_sampling_source = "ref"
    gt_plani_results: Dict = {
        "dx": {
            "nuth_offset": 0.997,
            "total_offset": 3.0,
            "unit_offset": "px",
            "total_bias_value": 0.0025,
            "unit_bias_value": "deg",
        },
        "dy": {
            "nuth_offset": 1.99,
            "total_offset": 5.0,
            "unit_offset": "px",
            "total_bias_value": 0.00416,
            "unit_bias_value": "deg",
        },
        "gdal_translate_bounds": {
            "ulx": 40.00083,
            "uly": 39.99834,
            "lrx": 40.83166,
            "lry": 39.16917,
        },
    }

    coregistration_ = coregistration.Coregistration(cfg["coregistration"])

    # Compute coregistration
    transform = coregistration_.compute_coregistration(sec, ref)
    # Get coregistration results
    demcompare_results = coregistration_.demcompare_results

    # Test that the output offsets and bias are the same as gt
    np.testing.assert_allclose(gt_xoff, transform.x_offset, rtol=1e-02)
    np.testing.assert_allclose(gt_yoff, transform.y_offset, rtol=1e-02)
    assert gt_sampling_source == coregistration_.sampling_source
    np.testing.assert_allclose(
        gt_plani_results["dx"]["nuth_offset"],
        demcompare_results["coregistration_results"]["dx"]["nuth_offset"],
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_plani_results["dx"]["total_bias_value"],
        demcompare_results["coregistration_results"]["dx"]["total_bias_value"],
        rtol=1e-02,
    )

    np.testing.assert_allclose(
        gt_plani_results["dy"]["nuth_offset"],
        demcompare_results["coregistration_results"]["dy"]["nuth_offset"],
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_plani_results["dy"]["total_bias_value"],
        demcompare_results["coregistration_results"]["dy"]["total_bias_value"],
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_plani_results["gdal_translate_bounds"]["ulx"],
        demcompare_results["coregistration_results"]["gdal_translate_bounds"][
            "ulx"
        ],
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        gt_plani_results["gdal_translate_bounds"]["uly"],
        demcompare_results["coregistration_results"]["gdal_translate_bounds"][
            "uly"
        ],
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        gt_plani_results["gdal_translate_bounds"]["lrx"],
        demcompare_results["coregistration_results"]["gdal_translate_bounds"][
            "lrx"
        ],
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        gt_plani_results["gdal_translate_bounds"]["lry"],
        demcompare_results["coregistration_results"]["gdal_translate_bounds"][
            "lry"
        ],
        rtol=1e-03,
    )


def test_compute_coregistration_with_default_coregistration_srtm_sampling_ref():
    """
    Test the compute_coregistration function
    Input data:
    - default coregistration without input configuration
    - input DEMs present in "srtm_test_data" test root data directory
    - sampling value dem : "sec"
    - coregistration method : "Nuth et kaab"
    - non-zero initial disparity
    Validation data:
    - Manually computed gt_xoff, gt_yoff
    Validation process:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Checked parameters on the demcompare_results output dict are:
        - output Transform has the correct offsets
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("srtm_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Load dems
    ref = load_dem(cfg["input_ref"]["path"])
    sec = load_dem(cfg["input_sec"]["path"])

    # Define ground truth offsets
    gt_xoff = 3.0
    gt_yoff = 5.0
    # Create coregistration with default configuration
    coregistration_ = coregistration.Coregistration()

    # Compute coregistration
    transform = coregistration_.compute_coregistration(sec, ref)

    # Test that the output offsets and bias are the same as gt
    np.testing.assert_allclose(gt_xoff, transform.x_offset, rtol=1e-02)
    np.testing.assert_allclose(gt_yoff, transform.y_offset, rtol=1e-02)
