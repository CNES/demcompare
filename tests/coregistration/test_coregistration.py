#!/usr/bin/env python
# coding: utf8
# Disable the protected-access to test the functions
# pylint:disable=protected-access
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
This module contains functions to test all the
methods in the coregistration step.
"""
# pylint:disable=too-many-lines
import glob

# Standard imports
import os
from tempfile import TemporaryDirectory
from typing import Dict

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


def test_coregistration_save_optional_outputss():
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
