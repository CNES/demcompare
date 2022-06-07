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
This module contains functions to test all the methods in
img_tools module.
- crop_rasterio_source_with_roi is tested in test_dem_tools with reproject_dems
"""

# Standard imports
import os

# Third party imports
import numpy as np
import pytest

# Demcompare imports
from demcompare import dem_tools, img_tools
from demcompare.initialization import read_config_file

# Tests helpers
from .helpers import demcompare_test_data_path


@pytest.mark.unit_tests
def test_convert_pix_to_coord():
    """
    Test convert_pix_to_coord function
    Makes the conversion from pix to coord for
    different pixel values and tests the obtained
    coordinates.
    """
    # Define transformation
    trans = np.array(
        [
            5.962550e05,
            5.000000e02,
            0.000000e00,
            5.099745e06,
            0.000000e00,
            -5.000000e02,
        ]
    )

    # Positive y and negative x ----------------------------
    # Define pixellic points
    y_pix = 5.1
    x_pix = -4.35
    # Convert to coords
    x_coord, y_coord = img_tools.convert_pix_to_coord(trans, y_pix, x_pix)
    # Define ground truth coords
    x_coord_gt = 594080
    y_coord_gt = 5097195

    np.testing.assert_allclose(x_coord_gt, x_coord, rtol=1e-03)
    np.testing.assert_allclose(y_coord_gt, y_coord, rtol=1e-03)

    # Positive y and positive x ----------------------------
    # Define pixellic points
    y_pix = 5.1
    x_pix = 4.35
    # Convert to coords
    x_coord, y_coord = img_tools.convert_pix_to_coord(trans, y_pix, x_pix)
    # Define ground truth coords
    x_coord_gt = 598430
    y_coord_gt = 5097195

    np.testing.assert_allclose(x_coord_gt, x_coord, rtol=1e-03)
    np.testing.assert_allclose(y_coord_gt, y_coord_gt, rtol=1e-03)

    # Negative y and negative x ---------------------------
    # Define pixellic points
    y_pix = -5.1
    x_pix = -4.35
    # Convert to coords
    x_coord, y_coord = img_tools.convert_pix_to_coord(trans, y_pix, x_pix)
    # Define ground truth coords
    x_coord_gt = 594080
    y_coord_gt = 5102295

    np.testing.assert_allclose(x_coord_gt, x_coord, rtol=1e-03)
    np.testing.assert_allclose(y_coord_gt, y_coord_gt, rtol=1e-03)

    # Negative y and positive x ---------------------------
    # Define pixellic points
    y_pix = 5.1
    x_pix = -4.35
    # Convert to coords
    x_coord, y_coord = img_tools.convert_pix_to_coord(trans, y_pix, x_pix)
    # Define ground truth coords
    x_coord_gt = 594080
    y_coord_gt = 5097195

    np.testing.assert_allclose(x_coord_gt, x_coord, rtol=1e-03)
    np.testing.assert_allclose(y_coord_gt, y_coord_gt, rtol=1e-03)


@pytest.mark.unit_tests
def test_compute_gdal_translate_bounds():
    """
    Test the compute_offset_bounds function
    Loads the DEMS present in "strm_test_data" and "gironde_test_data"
    test root data directory and computes the coordinate offset
    bounds for a given pixellic offset to test the resulting
    bounds.
    TODO: to move with refactoring coregistration
    """

    # Test with "strm_test_data" input dem
    # Get "gironde_test_data" test
    # root data directory absolute path
    test_data_path = demcompare_test_data_path("strm_test_data")
    # Load "gironde_test_data" demcompare
    # config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    dem = dem_tools.load_dem(cfg["input_dem_to_align"]["path"])

    # Pixellic offset
    dy_px = 0.00417
    dx_px = 0.0025

    # Define the ground truth reprojected values
    gt_ulx = 40.00000
    gt_uly = 40.00000
    gt_lrx = 40.83083
    gt_lry = 39.17083

    ulx, uly, lrx, lry = img_tools.compute_gdal_translate_bounds(
        -dy_px, dx_px, dem["image"].shape, dem["georef_transform"].data
    )
    # Test that the reprojected offsets are the same as ground_truth
    np.testing.assert_allclose(ulx, gt_ulx, rtol=1e-04)
    np.testing.assert_allclose(uly, gt_uly, rtol=1e-04)
    np.testing.assert_allclose(lrx, gt_lrx, rtol=1e-04)
    np.testing.assert_allclose(lry, gt_lry, rtol=1e-04)

    # Test with "gironde_test_data" input dem
    # Get "gironde_test_data" test
    # root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare
    # config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    dem = dem_tools.load_dem(cfg["input_dem_to_align"]["path"])

    # Pixellic offset
    dy_nuth = 0.41903
    dx_nuth = -1.43664

    # Define the ground truth reprojected values
    gt_ulx = 599536.68
    gt_uly = 5099954.515
    gt_lrx = 708536.68
    gt_lry = 4990954.515

    ulx, uly, lrx, lry = img_tools.compute_gdal_translate_bounds(
        -dy_nuth, dx_nuth, dem["image"].shape, dem["georef_transform"].data
    )

    # Test that the reprojected offsets are the same as ground_truth
    np.testing.assert_allclose(ulx, gt_ulx, rtol=1e-04)
    np.testing.assert_allclose(uly, gt_uly, rtol=1e-04)
    np.testing.assert_allclose(lrx, gt_lrx, rtol=1e-04)
    np.testing.assert_allclose(lry, gt_lry, rtol=1e-04)
