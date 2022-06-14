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

# Standard imports
import os

# Third party imports
import numpy as np
import pytest

# Demcompare imports
from demcompare import coregistration
from demcompare.dem_tools import SamplingSourceParameter, load_dem
from demcompare.initialization import read_config_file

# Tests helpers
from tests.helpers import demcompare_test_data_path


@pytest.mark.unit_tests
def test_compute_coregistration_with_gironde_test_data_sampling_dem():
    """
    Test the compute_coregistration function:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Tests that the output Transform has the correct offsets
    - Tests that the considered sampling source is correct
    - Test that the offsets, bias and and gdal_translate_bounds
      on the demcompare_results output dict are corrects

    Test configuration:
    - "gironde_test_data" input DEMs
    - sampling value dem
    - coregistration method Nuth et kaab
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
    gt_plani_results = {
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
    Test the compute_coregistration function:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Tests that the output Transform has the correct offsets
    - Tests that the considered sampling source is correct
    - Test that the offsets, bias and and gdal_translate_bounds
      on the demcompare_results output dict are corrects

    Test configuration:
    - "gironde_test_data" input DEMs
    - sampling value ref
    - coregistration method Nuth et kaab
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
    gt_plani_results = {
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


def test_compute_coregistration_with_strm_sampling_dem_and_initial_disparity():
    """
    Test the compute_coregistration function:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Tests that the output Transform has the correct offsets
    - Tests that the considered sampling source is correct
    - Test that the offsets, bias and and gdal_translate_bounds
      on the demcompare_results output dict are corrects

    Test configuration:
    - "strm_test_data" input DEMs
    - sampling value dem
    - coregistration method Nuth et kaab
    - non-zero initial disparity
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("strm_test_data")
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
    gt_plani_results = {
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


def test_compute_coregistration_with_strm_sampling_ref_and_initial_disparity():
    """
    Test the compute_coregistration function:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Tests that the output Transform has the correct offsets
    - Tests that the considered sampling source is correct
    - Test that the offsets, bias and and gdal_translate_bounds
      on the demcompare_results output dict are corrects

    Test configuration:
    - "strm_test_data" input DEMs
    - sampling value ref
    - coregistration method Nuth et kaab
    - non-zero initial disparity
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("strm_test_data")
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
    gt_plani_results = {
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


def test_compute_coregistration_gironde_sampling_dem_and_initial_disparity():
    """
    Test the compute_coregistration function:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Tests that the output Transform has the correct offsets
    - Tests that the considered sampling source is correct

    Test configuration:
    - "gironde_test_data" input DEMs
    - sampling value dem
    - coregistration method Nuth et kaab
    - non-zero initial disparity
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
    Test the compute_coregistration function:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Tests that the output Transform has the correct offsets
    - Tests that the considered sampling source is correct

    Test configuration:
    - "gironde_test_data" input DEMs
    - sampling value ref
    - coregistration method Nuth et kaab
    - non-zero initial disparity
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


def test_compute_coregistration_with_default_coregistration_strm_sampling_dem():
    """
    Test the compute_coregistration function:
    - Loads the data present in the test root data directory
    - Creates a coregistration object and does compute_coregistration
    - Tests that the output Transform has the correct offsets

    Test configuration:
    - default coregistration without input configuration
    - "strm_test_data" input DEMs
    - sampling value dem
    - coregistration method Nuth et kaab
    - non-zero initial disparity
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("strm_test_data")
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
