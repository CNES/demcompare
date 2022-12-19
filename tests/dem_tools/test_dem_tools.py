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
This module contains functions to test some methods in
dem_tools module.
"""
# pylint:disable = duplicate-code
# Standard imports
import os

# Third party imports
import numpy as np
import pytest
import rasterio

# Demcompare imports
from demcompare import dem_tools
from demcompare.dem_tools import DEFAULT_NODATA
from demcompare.helpers_init import read_config_file

# Tests helpers
from tests.helpers import demcompare_test_data_path

# Force protected access to test protected functions
# pylint:disable=protected-access


@pytest.mark.unit_tests
def test_translate_dem_no_offset(load_gironde_dem):
    """
    Test the translate_dem function without input offset.
    Input data:
    - Sec dem from the "load_gironde_dem" fixture.
    Validation data:
    - The input dem's georef_transform's values at position
      0 and 3: gt_offset_coord_x,
      gt_offset_coord_y
    Validation process:
    - Apply a zero pixellic offset (0, 0)
      to the dem using the translate_dem function
    - Check that the translated dem's georef_transform
      positions 0 and 3 have not changed
    - Checked function : dem_tools's translate_dem
    """
    # When translate_dem, the values at position 0 and 3 of the
    # transform are modified
    # by the offset coordinates x and y
    # Original dem transform 0 and 3 position values before translate_dem :
    # trans[0] = 600255
    # trans[3] = 5099745

    dem, _ = load_gironde_dem
    # No offset ------------------------------------
    y_off_pix = 0
    x_off_pix = 0
    # Define ground truth values
    gt_offset_coord_x = 600255
    gt_offset_coord_y = 5099745

    transformed_dem = dem_tools.translate_dem(dem, x_off_pix, y_off_pix)
    # Verify that the transform of the transformed
    # dem has the ground truth values
    np.testing.assert_allclose(
        gt_offset_coord_x,
        transformed_dem["georef_transform"].data[0],
        atol=1e-02,
    )
    np.testing.assert_allclose(
        gt_offset_coord_y,
        transformed_dem["georef_transform"].data[3],
        atol=1e-02,
    )


@pytest.mark.unit_tests
def test_translate_dem_original_transform(load_gironde_dem):
    """
    Test that the dem given to the
    translate function does not have its
    georeference_transform modified,
    only the returned dem does.
    Input data:
    - Sec dem from the "load_gironde_dem" fixture.
    Validation data:
    - The input dem's georef_transform's values at position
      0 and 3: gt_offset_coord_x,
      gt_offset_coord_y
    Validation process:
    - Apply a pixellic offset
      to the dem using the translate_dem function
    - Check that the original dem's georef_transform
      positions 0 and 3 have not changed
    - Checked function : dem_tools's translate_dem
    """
    # When translate_dem, the values at position 0 and 3 of the
    # transform are modified
    # by the offset coordinates x and y
    # Original dem transform 0 and 3 position values before translate_dem :
    # trans[0] = 600255
    # trans[3] = 5099745

    dem, _ = load_gironde_dem
    # Offset ------------------------------------
    y_off_pix = 5
    x_off_pix = 4
    # Define original dem offset values
    gt_offset_coord_x = 600255.0
    gt_offset_coord_y = 5099745.0

    _ = dem_tools.translate_dem(dem, x_off_pix, y_off_pix)
    # Verify that the transform of the transformed
    # dem has the ground truth values
    np.testing.assert_allclose(
        gt_offset_coord_x,
        dem["georef_transform"].data[0],
        atol=1e-02,
    )
    np.testing.assert_allclose(
        gt_offset_coord_y,
        dem["georef_transform"].data[3],
        atol=1e-02,
    )


@pytest.mark.unit_tests
def test_compute_dems_diff_custom_nodata():
    """
    Test compute_dems_diff function
    Input data:
    - Two manually created dems with custom nodata (-37, 99, 33)
      values
    Validation data:
    - Manually computed dem diff: diff_gt
    Validation process:
    - Create both dems
    - Compute the difference dem using the compute_dems_diff function
    - Check that the difference dem is the same as ground truth
    - Checked function : dem_tools's compute_dems_diff
    """
    # Create input datasets
    sec = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [-1, -37, 1],
            [1, 2, -37],
            [1, 1, -37],
        ],
        dtype=np.float32,
    )
    ref = np.array(
        [[3, 99, 3], [99, 2, 1], [99, 0, 1], [1, 1, 0], [1, 1, 99]],
        dtype=np.float32,
    )

    sec_dataset = dem_tools.create_dem(data=sec, nodata=-37)
    ref_dataset = dem_tools.create_dem(data=ref, nodata=99)

    # Define ground truth value
    diff_gt = np.array(
        [
            [3 - 1, np.nan, 3 - 1],
            [np.nan, 2 - 1, 1 - 1],
            [np.nan, np.nan, 1 - 1],
            [1 - 1, 1 - 2, np.nan],
            [1 - 1, 1 - 1, np.nan],
        ],
        dtype=np.float32,
    )

    diff_dataset = dem_tools.compute_dems_diff(ref_dataset, sec_dataset)
    # Test that the output difference is the same as ground_truth
    np.testing.assert_array_equal(diff_gt, diff_dataset["image"].data)

    # Create input datasets
    sec = np.array(
        [[1, 1, 1], [1, 1, 1], [-1, -33, 1], [1, 2, -33], [1, 1, -33]],
        dtype=np.float32,
    )
    ref = np.array(
        [
            [3, 3, 3],
            [1, 2, 1],
            [DEFAULT_NODATA, 0, 1],
            [1, 1, 0],
            [1, 1, DEFAULT_NODATA],
        ],
        dtype=np.float32,
    )

    sec_dataset = dem_tools.create_dem(data=sec, nodata=-33)
    ref_dataset = dem_tools.create_dem(data=ref)

    # Define ground truth value

    diff_gt = np.array(
        [
            [3 - 1, 3 - 1, 3 - 1],
            [1 - 1, 2 - 1, 1 - 1],
            [np.nan, np.nan, 1 - 1],
            [1 - 1, 1 - 2, np.nan],
            [1 - 1, 1 - 1, np.nan],
        ],
        dtype=np.float32,
    )
    diff_dataset = dem_tools.compute_dems_diff(ref_dataset, sec_dataset)
    # Test that the output difference is the same as ground_truth
    np.testing.assert_array_equal(diff_gt, diff_dataset["image"].data)


@pytest.mark.unit_tests
def test_compute_dem_diff_bounds_transform():
    """
    Test the compute_dem_diff output dem's bounds and transform
    Input data:
    - input DEMs present in "strm_test_data" test root data directory
    - Hand crafted data
    Validation data:
    - Manually computed ROI and BoundingBox obtained by rasterio
    - Hand crafted ground truth
    Validation process:
    - Load the dem with the load_dem function.
    - Create dems with create_dem function.
    - Testing datas are:
        - bounds
        - data
        - crs
    - Checked function: compute_dem_diff
    """
    # Get "srtm_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("srtm_test_data")
    # Load "srtm_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    from_dataset = dem_tools.load_dem(cfg["input_sec"]["path"])

    # Define data dem 1
    data_1 = np.array(
        [[1, 2, 3], [1, 4, 5], [-1, -32768, 6], [1, 7, -32768], [1, 8, -32768]],
        dtype=np.float32,
    )

    # Define data dem 2
    data_2 = np.array(
        [[1, 9, 8], [1, 7, 6], [-1, -32768, 5], [1, 4, -32768], [1, 3, -32768]],
        dtype=np.float32,
    )

    ground_truth = np.array(
        [
            [0.0, -7.0, -5.0],
            [0.0, -3.0, -1.0],
            [0.0, np.nan, 1.0],
            [0.0, 3.0, np.nan],
            [0.0, 5.0, np.nan],
        ],
        dtype=np.float32,
    )

    # Create dataset from "srtm_test_data" DSM and specific nodata value
    dem_1 = dem_tools.create_dem(
        data=data_1,
        transform=from_dataset.georef_transform.data,
        img_crs=from_dataset.crs,
        nodata=-32768,
        bounds=from_dataset.bounds,
    )

    # Create dataset from "srtm_test_data" DSM and specific nodata value
    dem_2 = dem_tools.create_dem(
        data=data_2,
        transform=from_dataset.georef_transform.data,
        img_crs=from_dataset.crs,
        nodata=-32768,
        bounds=from_dataset.bounds,
    )

    alti_dif = dem_tools.compute_dems_diff(dem_1, dem_2)

    np.testing.assert_allclose(
        alti_dif["image"].data,
        ground_truth,
        rtol=1e-02,
    )
    assert alti_dif.bounds == from_dataset.bounds
    assert alti_dif.crs == from_dataset.crs


@pytest.mark.unit_tests
def test_compute_waveform():
    """
    Test the compute_waveform function
    Input data:
    - A manually created input dem
    Validation data:
    - The manually computed dem's col and row waveforms:
      gt_col_waveform, gt_row_waveform
    Validation process:
    - Create the dem using the create_dem function
    - Compute dem's waveform using the compute_waveform function
    - Check that the obtained waveform is the same as ground truth
    - Checked function : dem_tools's compute_waveform
    """

    # Initialize input data
    data = np.array(
        [[-7.0, 3.0, 3.0], [1, 3.0, 1.0], [3.0, 1.0, 0.0]], dtype=np.float32
    )
    dem = dem_tools.create_dem(data=data)

    # Compute gt waveform
    mean_row = np.nanmean(data, axis=0)
    gt_row_waveform = data - mean_row

    mean_col = np.nanmean(data, axis=1)

    mean_col = np.transpose(
        np.ones((1, mean_col.size), dtype=np.float32) * mean_col
    )
    gt_col_waveform = data - mean_col

    # Obtain output waveform
    output_row_waveform, output_col_waveform = dem_tools.compute_waveform(dem)

    np.testing.assert_allclose(gt_col_waveform, output_col_waveform, rtol=1e-02)
    np.testing.assert_allclose(gt_row_waveform, output_row_waveform, rtol=1e-02)


@pytest.mark.unit_tests
def test_compute_dem_slope():
    """
    Test the compute_dem_slope function.
    Input data:
    - A manually created input dem
    Validation data:
    - The manually computed dem's slope
    Validation process:
    - Create the dem using the create_dem function
    - Compute the dem's slope using the compute_dem_slope function
    - Check that the obtained slope is the same as ground truth
    - Checked function: dem_tools's compute_dem_slope
    """

    # Generate dsm with the following data and
    # "gironde_test_data" DSM's georef and resolution
    data = np.array([[1, 0, 1], [1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    dem_dataset = dem_tools.create_dem(
        data=data,
        img_crs=rasterio.crs.CRS.from_epsg(32630),
    )
    # Compute dem's slope
    dem_dataset = dem_tools.compute_dem_slope(dem_dataset)
    # Slope filters are the following
    # conv_x = np.array([[-1, 0, 1],
    #                   [-2, 0, 2],
    #                   [-1, 0, 1]])
    # conv_y = np.array([[-1, -2, -1],
    #                   [ 0,  0,  0],
    #                   [ 1,  2,  1]])

    # Convolution result
    gx = np.array([[4, 0, -4], [2, -2, -4], [-2, -6, -4]])
    gy = np.array([[0, 0, 0], [6, 2, 0], [6, 2, 0]])
    # Dataset's absolute resolutions are 500, 500
    distx = np.abs(dem_dataset.xres)
    disty = np.abs(dem_dataset.yres)

    # Compute tan(slope) and aspect
    tan_slope = np.sqrt((gx / distx) ** 2 + (gy / disty) ** 2) / 8
    gt_slope = np.arctan(tan_slope) * 100

    output_slope = dem_tools.compute_dem_slope(dem_dataset)
    # Test that the output_slope is the same as ground_truth
    np.testing.assert_allclose(
        gt_slope,
        output_slope["ref_slope"].data[:, :],
        rtol=1e-02,
    )
