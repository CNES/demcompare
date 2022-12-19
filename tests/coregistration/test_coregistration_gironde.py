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
methods in the coregistration step with gironde datas.
"""
# pylint:disable = duplicate-code
# Standard imports
import os
from typing import Dict

# Third party imports
import numpy as np
import pytest

# Demcompare imports
from demcompare import coregistration
from demcompare.dem_tools import SamplingSourceParameter, load_dem
from demcompare.helpers_init import read_config_file

# Tests helpers
from tests.helpers import demcompare_test_data_path


@pytest.mark.unit_tests
def test_compute_coregistration_with_gironde_test_data_sampling_dem():
    """
    Test the compute_coregistration function
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    - sampling value : "sec"
    - coregistration method : "Nuth et kaab"
    Validation data:
    - Manually computed gt_xoff, gt_yoff,
      gt_sampling_source and gt_plani_results
    Validation process:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Checked parameters on the demcompare_results output dict are:
        - output Transform correct offsets
        - considered sampling source "sec"
        - offsets
        - bias
        - gdal_translate_bounds
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Load dems
    ref = load_dem(cfg["input_ref"]["path"])
    sec = load_dem(cfg["input_sec"]["path"])

    # Define ground truth offsets
    gt_xoff = -1.43664
    gt_yoff = -0.41903
    gt_sampling_source = "sec"
    gt_plani_results: Dict = {
        "dx": {
            "nuth_offset": -1.43664,
            "total_offset": -1.43664,
            "unit_offset": "px",
            "total_bias_value": -718.31907,
            "unit_bias_value": "m",
        },
        "dy": {
            "nuth_offset": -0.41903,
            "total_offset": -0.41903,
            "unit_offset": "px",
            "total_bias_value": -209.51619,
            "unit_bias_value": "m",
        },
        "gdal_translate_bounds": {
            "ulx": 599536.68093,
            "uly": 5099954.51619,
            "lrx": 708536.68093,
            "lry": 4990954.51619,
        },
    }

    # Initialize coregistration object
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
    assert (
        gt_plani_results["dx"]["unit_offset"]
        == demcompare_results["coregistration_results"]["dx"]["unit_offset"]
    )
    np.testing.assert_allclose(
        gt_plani_results["dx"]["total_bias_value"],
        demcompare_results["coregistration_results"]["dx"]["total_bias_value"],
        rtol=1e-02,
    )
    assert (
        gt_plani_results["dx"]["unit_bias_value"]
        == demcompare_results["coregistration_results"]["dx"]["unit_bias_value"]
    )

    np.testing.assert_allclose(
        gt_plani_results["dy"]["nuth_offset"],
        demcompare_results["coregistration_results"]["dy"]["nuth_offset"],
        rtol=1e-02,
    )
    assert (
        gt_plani_results["dy"]["unit_offset"]
        == demcompare_results["coregistration_results"]["dy"]["unit_offset"]
    )
    np.testing.assert_allclose(
        gt_plani_results["dy"]["total_bias_value"],
        demcompare_results["coregistration_results"]["dy"]["total_bias_value"],
        rtol=1e-02,
    )
    assert (
        gt_plani_results["dy"]["unit_bias_value"]
        == demcompare_results["coregistration_results"]["dy"]["unit_bias_value"]
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


def test_compute_coregistration_with_gironde_test_data_sampling_ref():
    """
    Test the compute_coregistration function
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    - sampling value : "ref"
    - coregistration method : "Nuth et kaab"
    Validation data:
    - Manually computed gt_xoff, gt_yoff,
      gt_sampling_source and gt_plani_results
    Validation process:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Checked parameters on the demcompare_results output dict are:
        - output Transform correct offsets
        - considered sampling source "ref"
        - offsets
        - bias
        - gdal_translate_bounds
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Modify cfg's sampling_source (default is "sec")
    cfg["coregistration"]["sampling_source"] = SamplingSourceParameter.REF.value

    # Load dems
    ref = load_dem(cfg["input_ref"]["path"])
    sec = load_dem(cfg["input_sec"]["path"])

    # Define ground truth offsets
    gt_xoff = -1.0864
    gt_yoff = -0.22552
    gt_sampling_source = "ref"
    gt_plani_results: Dict = {
        "dx": {
            "nuth_offset": -1.08642,
            "total_offset": -1.08642,
            "unit_offset": "px",
            "total_bias_value": -543.21172,
            "unit_bias_value": "m",
        },
        "dy": {
            "nuth_offset": -0.22552,
            "total_offset": -0.22552,
            "unit_offset": "px",
            "total_bias_value": -112.76041,
            "unit_bias_value": "m",
        },
        "gdal_translate_bounds": {
            "ulx": 599711.78828,
            "uly": 5099857.76041,
            "lrx": 708711.78828,
            "lry": 4990857.76041,
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
    assert (
        gt_plani_results["dx"]["unit_offset"]
        == demcompare_results["coregistration_results"]["dx"]["unit_offset"]
    )
    np.testing.assert_allclose(
        gt_plani_results["dx"]["total_bias_value"],
        demcompare_results["coregistration_results"]["dx"]["total_bias_value"],
        rtol=1e-02,
    )
    assert (
        gt_plani_results["dx"]["unit_bias_value"]
        == demcompare_results["coregistration_results"]["dx"]["unit_bias_value"]
    )
    np.testing.assert_allclose(
        gt_plani_results["dy"]["nuth_offset"],
        demcompare_results["coregistration_results"]["dy"]["nuth_offset"],
        rtol=1e-02,
    )
    assert (
        gt_plani_results["dy"]["unit_offset"]
        == demcompare_results["coregistration_results"]["dy"]["unit_offset"]
    )
    np.testing.assert_allclose(
        gt_plani_results["dy"]["total_bias_value"],
        demcompare_results["coregistration_results"]["dy"]["total_bias_value"],
        rtol=1e-02,
    )
    assert (
        gt_plani_results["dy"]["unit_bias_value"]
        == demcompare_results["coregistration_results"]["dy"]["unit_bias_value"]
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


def test_compute_coregistration_gironde_sampling_sec_and_initial_disparity():
    """
    Test the compute_coregistration function
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    - sampling value dem : "sec"
    - coregistration method : "Nuth et kaab"
    - non-zero initial disparity
    Validation data:
    - Manually computed gt_xoff, gt_yoff and gt_sampling_source
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
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Modify cfg's estimated_initial_shift
    cfg["coregistration"]["estimated_initial_shift_x"] = -0.5
    cfg["coregistration"]["estimated_initial_shift_y"] = 0.2

    # Load dems
    ref = load_dem(cfg["input_ref"]["path"])
    sec = load_dem(cfg["input_sec"]["path"])

    # Define ground truth offsets
    gt_xoff = -1.3065
    gt_yoff = -0.4796
    gt_sampling_source = "sec"

    coregistration_ = coregistration.Coregistration(cfg["coregistration"])

    # Compute coregistration
    transform = coregistration_.compute_coregistration(sec, ref)

    # Test that the output offsets and bias are the same as gt
    np.testing.assert_allclose(gt_xoff, transform.x_offset, rtol=1e-02)
    np.testing.assert_allclose(gt_yoff, transform.y_offset, rtol=1e-02)
    # Get coregistration results
    assert gt_sampling_source == coregistration_.sampling_source


def test_compute_coregistration_gironde_sampling_ref_and_initial_disparity():
    """
    Test the compute_coregistration function
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    - sampling value dem : "ref"
    - coregistration method : "Nuth et kaab"
    - non-zero initial disparity
    Validation data:
    - Manually computed gt_xoff, gt_yoff and gt_sampling_source
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
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Modify cfg's sampling_source (default is "sec")
    cfg["coregistration"]["sampling_source"] = SamplingSourceParameter.REF.value

    # Modify cfg's estimated_initial_shift
    cfg["coregistration"]["estimated_initial_shift_x"] = -0.5
    cfg["coregistration"]["estimated_initial_shift_y"] = 0.2

    # Load dems
    ref = load_dem(cfg["input_ref"]["path"])
    sec = load_dem(cfg["input_sec"]["path"])

    # Define ground truth offsets
    gt_xoff = -1.0509
    gt_yoff = -0.2321
    gt_sampling_source = "ref"

    coregistration_ = coregistration.Coregistration(cfg["coregistration"])

    # Compute coregistration
    transform = coregistration_.compute_coregistration(sec, ref)

    # Test that the output offsets and bias are the same as gt
    np.testing.assert_allclose(gt_xoff, transform.x_offset, rtol=1e-02)
    np.testing.assert_allclose(gt_yoff, transform.y_offset, rtol=1e-02)
    assert gt_sampling_source == coregistration_.sampling_source


def test_coregistration_with_same_dems():
    """
    Test the coregistration with same dem as entry
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    - input_ref and input_sec are the same
    - coregistration method : "Nuth et kaab"
    Validation data:
    - None
    Validation process:
    - Load input/test_config.json
    - Replace input_sec by input_ref in config
    - Test that ValueError is raised
    Validation data:
    - Manually computed gt_xoff, gt_yoff
    Validation process:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Checked parameters on the demcompare_results output dict are:
       - output Transform correct offsets
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Same dem for second and reference
    cfg["input_sec"] = cfg["input_ref"]

    # Load dems
    ref = load_dem(cfg["input_ref"]["path"])
    sec = load_dem(cfg["input_sec"]["path"])

    # Define ground truth offsets
    gt_xoff = 0.0
    gt_yoff = 0.0

    # Initialize coregistration object
    coregistration_ = coregistration.Coregistration(cfg["coregistration"])
    # Compute coregistration
    transform = coregistration_.compute_coregistration(sec, ref)

    # Test that the output offsets and bias are the same as gt
    np.testing.assert_allclose(gt_xoff, transform.x_offset, rtol=1e-02)
    np.testing.assert_allclose(gt_yoff, transform.y_offset, rtol=1e-02)
