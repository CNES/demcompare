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
dem_tools module.
TODO:
- add test_save_dem
- add test_copy_dem
- check wrong opening of DEM test_load_wrong_dem.
- check reproject_dems with CRS ref and dem_to_align inversed.
- check CRS in compute_dems_diff ?
"""

# Standard imports
import os

# Third party imports
import numpy as np
import pytest
import rasterio

# Demcompare imports
from demcompare import dataset_tools, dem_tools
from demcompare.initialization import read_config_file

# Tests helpers
from .helpers import demcompare_path, demcompare_test_data_path

# Force protected access to test protected functions
# pylint:disable=protected-access


@pytest.mark.unit_tests
def test_load_dem():
    """
    Test the load_dem function
    Loads the data present in "strm_test_data" test root
    data directory and tests the loaded DEM Dataset.

    """
    # Get "strm_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("strm_test_data")
    # Load "strm_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    dem = dem_tools.load_dem(cfg["input_ref"]["path"])

    # Define ground truth values
    gt_no_data = -32768.0
    gt_xres = 0.000833333333333333
    gt_yres = -0.0008333333333333334
    gt_plani_unit = "deg"
    gt_zunit = "m"
    gt_georef = "EPSG:4326"
    gt_shape = (1000, 1000)
    gt_transform = np.array(
        [
            4.00000000e01,
            8.33333333e-04,
            0.00000000e00,
            4.00000000e01,
            0.00000000e00,
            -8.33333333e-04,
        ]
    )
    # Test that the loaded dem has the groud truth values
    assert gt_no_data == dem.attrs["no_data"]
    np.testing.assert_allclose(gt_xres, dem.attrs["xres"], rtol=1e-02)
    np.testing.assert_allclose(gt_yres, dem.attrs["yres"], rtol=1e-02)
    assert gt_plani_unit == dem.attrs["plani_unit"]
    assert gt_zunit == dem.attrs["zunit"]
    assert gt_georef == dem.attrs["crs"]
    assert gt_shape == dem["image"].shape
    np.testing.assert_allclose(gt_transform, dem.georef_transform, rtol=1e-02)


@pytest.mark.unit_tests
def test_translate_dem():
    """
    Test the translate_dem function
    Loads the DEMS present in "gironde_test_data" test root
    data directory and makes the DEM translation for
    different pixel values and tests the resulting
    transform.
    """
    # Test with "gironde_test_data" test root input DEMs
    # and sampling value ref
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    dem = dem_tools.load_dem(cfg["input_dem_to_align"]["path"])

    # When translate_dem, the values at position 0 and 3 of the
    # transform are modified
    # by the offset coordinates x and y
    # Original dem transform 0 and 3 position values before translate_dem :
    # trans[0] = 600255
    # trans[3] = 5099745

    # Negative y and positive x -----------------------------
    y_off_pix = -5
    x_off_pix = 4
    # Define ground truth values
    gt_offset_coord_x = 602255
    gt_offset_coord_y = 5102245

    transformed_dem = dem_tools.translate_dem(dem, x_off_pix, y_off_pix)
    # Verify that the transform of the transformed
    # dem has the ground truth values
    np.testing.assert_allclose(
        gt_offset_coord_x,
        transformed_dem["georef_transform"].data[0],
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_offset_coord_y,
        transformed_dem["georef_transform"].data[3],
        rtol=1e-02,
    )

    # Negative y and negative x --------------------------
    y_off_pix = -5
    x_off_pix = -4
    # Define ground truth values
    gt_offset_coord_x = 600255
    gt_offset_coord_y = 5104745

    transformed_dem = dem_tools.translate_dem(dem, x_off_pix, y_off_pix)
    # Verify that the transform of the transformed
    # dem has the ground truth values
    np.testing.assert_allclose(
        gt_offset_coord_x,
        transformed_dem["georef_transform"].data[0],
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_offset_coord_y,
        transformed_dem["georef_transform"].data[3],
        rtol=1e-02,
    )

    # Positive y and negative x ----------------------------
    y_off_pix = 5
    x_off_pix = -4
    # Define ground truth values
    gt_offset_coord_x = 598255
    gt_offset_coord_y = 5102245

    transformed_dem = dem_tools.translate_dem(dem, x_off_pix, y_off_pix)
    # Verify that the transform of the transformed
    # dem has the ground truth values
    np.testing.assert_allclose(
        gt_offset_coord_x,
        transformed_dem["georef_transform"].data[0],
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_offset_coord_y,
        transformed_dem["georef_transform"].data[3],
        rtol=1e-02,
    )

    # Positive y and positive x ----------------------------
    y_off_pix = 5
    x_off_pix = -4
    # Define ground truth values
    gt_offset_coord_x = 596255
    gt_offset_coord_y = 5099745

    transformed_dem = dem_tools.translate_dem(dem, x_off_pix, y_off_pix)
    # Verify that the transform of the transformed
    # dem has the ground truth values
    np.testing.assert_allclose(
        gt_offset_coord_x,
        transformed_dem["georef_transform"].data[0],
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_offset_coord_y,
        transformed_dem["georef_transform"].data[3],
        rtol=1e-02,
    )

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
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_offset_coord_y,
        transformed_dem["georef_transform"].data[3],
        rtol=1e-02,
    )


@pytest.mark.unit_tests
def test_reproject_dems():
    """
    Test the reproject_dems function
    Loads the DEMS present in "gironde_test_data" test root
    data directory and reprojects them to test the
    obtained reprojected DEMs.

    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load original dems
    ref_orig = dem_tools.load_dem(cfg["input_ref"]["path"])
    dem_to_align_orig = dem_tools.load_dem(cfg["input_dem_to_align"]["path"])

    # Reproject dems with sampling value dem  -------------------------------
    (
        reproj_dem_to_align,
        reproj_ref,
        adapting_factor,
    ) = dem_tools.reproject_dems(
        dem_to_align_orig,
        ref_orig,
        sampling_source=dem_tools.SamplingSourceParameter.DEM_TO_ALIGN.value,
    )

    # Define ground truth values
    gt_intersection_roi = (600255.0, 4990745.0, 689753.076, 5090117.757)
    # Since sampling value is "dem_to_align",
    # output resolutions is "dem_to_align"'s resolution
    gt_output_yres = -500.00
    gt_output_xres = 500.00
    gt_output_shape = (199, 179)
    gt_adapting_factor = (1.0, 1.0)
    gt_output_trans = np.array(
        [
            6.002550e05,
            5.000000e02,
            0.000000e00,
            5.090245e06,
            0.000000e00,
            -5.000000e02,
        ]
    )

    # Test that both the output dems have the same gt values
    # Tests dems shape
    np.testing.assert_array_equal(
        reproj_dem_to_align["image"].shape, reproj_ref["image"].shape
    )
    np.testing.assert_allclose(
        reproj_dem_to_align["image"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["image"].shape, gt_output_shape, rtol=1e-03
    )
    # Tests dems resolution
    np.testing.assert_allclose(
        reproj_dem_to_align.yres, gt_output_yres, rtol=1e-02
    )
    np.testing.assert_allclose(reproj_ref.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(
        reproj_dem_to_align.xres, gt_output_xres, rtol=1e-02
    )
    np.testing.assert_allclose(reproj_ref.xres, gt_output_xres, rtol=1e-02)
    # Tests dems bounds
    np.testing.assert_allclose(
        reproj_dem_to_align.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    # Test adapting_factor
    np.testing.assert_allclose(adapting_factor, gt_adapting_factor, rtol=1e-02)
    # Test dems transform
    np.testing.assert_allclose(
        reproj_dem_to_align.georef_transform, gt_output_trans, rtol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.georef_transform, gt_output_trans, rtol=1e-02
    )

    # Reproject dems with sampling value ref --------------------------------

    (
        reproj_dem_to_align,
        reproj_ref,
        adapting_factor,
    ) = dem_tools.reproject_dems(
        dem_to_align_orig,
        ref_orig,
        sampling_source=dem_tools.SamplingSourceParameter.REF.value,
    )

    # Define ground truth values
    gt_intersection_roi = (-1.726, 45.039, -0.602, 45.939)
    # Since sampling value is "ref", output resolutions is "ref"'s resolution
    gt_output_yres = -0.0010416
    gt_output_xres = 0.0010416
    gt_output_shape = (865, 1079)
    gt_adapting_factor = (0.199451, 0.190893)
    gt_output_trans = np.array(
        [
            -1.72680447e00,
            1.04166667e-03,
            0.00000000e00,
            4.59394826e01,
            0.00000000e00,
            -1.04166667e-03,
        ]
    )
    # Test that both the output dems have the same gt values
    # Tests dems shape
    np.testing.assert_array_equal(
        reproj_dem_to_align["image"].shape, reproj_ref["image"].shape
    )
    np.testing.assert_allclose(
        reproj_dem_to_align["image"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["image"].shape, gt_output_shape, rtol=1e-03
    )
    # Tests dems resolution
    np.testing.assert_allclose(
        reproj_dem_to_align.yres, gt_output_yres, rtol=1e-02
    )
    np.testing.assert_allclose(reproj_ref.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(
        reproj_dem_to_align.xres, gt_output_xres, rtol=1e-02
    )
    np.testing.assert_allclose(reproj_ref.xres, gt_output_xres, rtol=1e-02)
    # Tests dems bounds
    np.testing.assert_allclose(
        reproj_dem_to_align.attrs["bounds"], gt_intersection_roi, rtol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.attrs["bounds"], gt_intersection_roi, rtol=1e-02
    )
    # Test adapting_factor
    np.testing.assert_allclose(adapting_factor, gt_adapting_factor, rtol=1e-02)
    # Test dems transform
    np.testing.assert_allclose(
        reproj_dem_to_align.georef_transform, gt_output_trans, rtol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.georef_transform, gt_output_trans, rtol=1e-02
    )

    # Reproject dems with sampling value dem and initial disparity -------------

    (
        reproj_dem_to_align,
        reproj_ref,
        adapting_factor,
    ) = dem_tools.reproject_dems(
        dem_to_align_orig,
        ref_orig,
        sampling_source=dem_tools.SamplingSourceParameter.DEM_TO_ALIGN.value,
        initial_shift_x=2,
        initial_shift_y=-3,
    )

    # Define ground truth values
    gt_intersection_roi = (600255.0, 4990745.0, 689753.076, 5090117.757)
    # Since sampling value is "dem_to_align",
    # output resolutions is "dem_to_align"'s resolution
    gt_output_yres = -500.00
    gt_output_xres = 500.00
    gt_output_shape = (199, 179)
    gt_adapting_factor = (1.0, 1.0)
    gt_output_trans = np.array(
        [
            6.012550e05,
            5.000000e02,
            0.000000e00,
            5.091745e06,
            0.000000e00,
            -5.000000e02,
        ]
    )

    # Test that both the output dems have the same gt values
    # Tests dems shape
    np.testing.assert_array_equal(
        reproj_dem_to_align["image"].shape, reproj_ref["image"].shape
    )
    np.testing.assert_allclose(
        reproj_dem_to_align["image"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["image"].shape, gt_output_shape, rtol=1e-03
    )
    # Tests dems resolution
    np.testing.assert_allclose(
        reproj_dem_to_align.yres, gt_output_yres, rtol=1e-02
    )
    np.testing.assert_allclose(reproj_ref.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(
        reproj_dem_to_align.xres, gt_output_xres, rtol=1e-02
    )
    np.testing.assert_allclose(reproj_ref.xres, gt_output_xres, rtol=1e-02)
    # Tests dems bounds
    np.testing.assert_allclose(
        reproj_dem_to_align.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    np.testing.assert_allclose(adapting_factor, gt_adapting_factor, rtol=1e-02)
    # Test adapting_factor
    np.testing.assert_allclose(adapting_factor, gt_adapting_factor, rtol=1e-02)
    # Test dems transform
    np.testing.assert_allclose(
        reproj_dem_to_align.georef_transform, gt_output_trans, rtol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.georef_transform, gt_output_trans, rtol=1e-02
    )


@pytest.mark.unit_tests
def test_compute_dems_diff():
    """
    Test compute_dems_diff function
    Creates two DEM datasets and computes its altitudes difference
    to test the obtained difference Dataset.
    """
    # Create input datasets
    dem_to_align = np.array(
        [[1, 1, 1], [1, 1, 1], [-1, -9999, 1], [1, 2, -9999], [1, 1, -9999]],
        dtype=np.float32,
    )
    ref = np.array(
        [[3, 3, 3], [1, 2, 1], [1, 0, 1], [1, 1, 0], [1, 1, 2]],
        dtype=np.float32,
    )

    dem_to_align_dataset = dem_tools.create_dem(data=dem_to_align)
    ref_dataset = dem_tools.create_dem(data=ref)

    # Define ground truth value
    diff_gt = np.array(
        [
            [3 - 1, 3 - 1, 3 - 1],
            [1 - 1, 2 - 1, 1 - 1],
            [1 + 1, np.nan, 1 - 1],
            [1 - 1, 1 - 2, np.nan],
            [1 - 1, 1 - 1, np.nan],
        ],
        dtype=np.float32,
    )

    diff_dataset = dem_tools.compute_dems_diff(
        ref_dataset, dem_to_align_dataset
    )
    # Test that the output difference is the same as ground_truth
    np.testing.assert_array_equal(diff_gt, diff_dataset["image"].data)

    # Create input datasets
    dem_to_align = np.array(
        [[1, 1, 1], [1, 1, 1], [-1, -33, 1], [1, 2, -33], [1, 1, -33]],
        dtype=np.float32,
    )
    ref = np.array(
        [[3, 3, 3], [1, 2, 1], [-9999, 0, 1], [1, 1, 0], [1, 1, -9999]],
        dtype=np.float32,
    )

    dem_to_align_dataset = dem_tools.create_dem(data=dem_to_align, no_data=-33)
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
    diff_dataset = dem_tools.compute_dems_diff(
        ref_dataset, dem_to_align_dataset
    )
    # Test that the output difference is the same as ground_truth
    np.testing.assert_array_equal(diff_gt, diff_dataset["image"].data)


@pytest.mark.unit_tests
def test_create_dem():
    """
    Test the _create_dem function
    Creates a dem with the data present in
    "gironde_test_data" test root data directory and tests
    the obtained DEM Dataset.
    """

    # Test with "gironde_test_data" input dem
    # Get "gironde_test_data" test
    # root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare
    # config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Define data
    data = np.ones((1000, 1000))

    # Modify original bounds, trans and nodata values
    bounds_dem = (
        600250,
        4990000,
        709200,
        5090000,
    )
    trans = np.array([700000, 600, 0, 1000000, 0, -600])
    nodata = -32768
    # Create dataset from the gironde_test_data
    # DSM with the defined data, bounds,
    # transform and nodata values
    dataset = dem_tools.create_dem(
        data=data,
        img_crs=rasterio.crs.CRS.from_epsg(32630),
        transform=trans,
        input_img=cfg["input_dem_to_align"]["path"],
        bounds=bounds_dem,
        no_data=nodata,
    )
    # Define the ground truth values
    # The created dataset should have the gironde_test_data DSM georef
    gt_img_crs = "EPSG:32630"
    gt_zunit = "m"
    gt_no_data = -32768
    gt_plani_unit = "m"

    # Test that the created dataset has the ground truth values
    np.testing.assert_allclose(trans, dataset.georef_transform, rtol=1e-02)
    np.testing.assert_allclose(gt_no_data, dataset.attrs["no_data"], rtol=1e-02)
    assert gt_plani_unit == dataset.attrs["plani_unit"]
    assert gt_zunit == dataset.attrs["zunit"]
    np.testing.assert_allclose(bounds_dem, dataset.attrs["bounds"], rtol=1e-02)
    assert gt_img_crs == dataset.attrs["crs"]

    # Test with geoid_georef set to True ---------------

    # Define geoid path
    geoid_path = demcompare_path("geoid/egm96_15.gtx")
    # Get geoid offset of the dataset
    output_arr_offset = dataset_tools._get_geoid_offset(dataset, geoid_path)
    # Add the geoid offset to the dataset
    gt_data_with_offset = dataset["image"].data + output_arr_offset

    # Compute the dataset with the geoid_georef parameter set to True
    output_dataset_with_offset = dem_tools.create_dem(
        data=data,
        img_crs=rasterio.crs.CRS.from_epsg(32630),
        transform=trans,
        input_img=cfg["input_dem_to_align"]["path"],
        bounds=bounds_dem,
        no_data=nodata,
        geoid_georef=True,
    )

    # Test that the output_dataset_with_offset has
    # the same values as the gt_dataset_with_offset
    np.testing.assert_allclose(
        output_dataset_with_offset["image"].data,
        gt_data_with_offset,
        rtol=1e-02,
    )
