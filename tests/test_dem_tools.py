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
"""
# pylint:disable = too-many-lines
# pylint:disable = duplicate-code
# Standard imports
import os
from tempfile import TemporaryDirectory
from typing import Dict

# Third party imports
import numpy as np
import pytest
import rasterio
import xarray as xr

# Demcompare imports
from demcompare import dataset_tools, dem_tools, load_input_dems
from demcompare.dem_tools import DEFAULT_NODATA
from demcompare.helpers_init import (
    compute_initialization,
    mkdir_p,
    read_config_file,
    save_config_file,
)

# Tests helpers
from .helpers import demcompare_path, demcompare_test_data_path, temporary_dir

# Force protected access to test protected functions
# pylint:disable=protected-access


@pytest.fixture(name="load_gironde_dem")
def fixture_load_gironde_dem():
    """
    Fixture to initialize the gironde dem for tests
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load original dems
    ref_orig = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec_orig = dem_tools.load_dem(cfg["input_sec"]["path"])

    return sec_orig, ref_orig


@pytest.fixture(name="initialize_dems_to_fuse")
def fixture_initialize_dems_to_fuse():
    """
    Fixture to initialize two dems to be fused
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")

    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Initialize sec and ref, necessary for StatsProcessing creation
    sec = dem_tools.load_dem(
        cfg["input_sec"]["path"],
        classification_layers=(cfg["input_sec"]["classification_layers"]),
    )
    ref = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec, ref, _ = dem_tools.reproject_dems(sec, ref)

    # Compute slope and add it as a classification_layer
    ref = dem_tools.compute_dem_slope(ref)
    sec = dem_tools.compute_dem_slope(sec)
    return ref, sec


@pytest.mark.unit_tests
def test_load_dem():
    """
    Test the load_dem function
    Loads the data present in "srtm_test_data" test root
    data directory and tests the loaded DEM Dataset.

    """
    # Get "srtm_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("srtm_test_data")
    # Load "srtm_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    dem = dem_tools.load_dem(cfg["input_ref"]["path"])

    # Define ground truth values
    gt_nodata = -32768.0
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
    assert gt_nodata == dem.attrs["nodata"]
    np.testing.assert_allclose(gt_xres, dem.attrs["xres"], rtol=1e-02)
    np.testing.assert_allclose(gt_yres, dem.attrs["yres"], rtol=1e-02)
    assert gt_plani_unit == dem.attrs["plani_unit"]
    assert gt_zunit == dem.attrs["zunit"]
    assert gt_georef == dem.attrs["crs"]
    assert gt_shape == dem["image"].shape
    np.testing.assert_allclose(gt_transform, dem.georef_transform, rtol=1e-02)


@pytest.mark.unit_tests
def test_translate_dem_pos_x_neg_y(load_gironde_dem):
    """
    Test the translate_dem function with positive x
    and negative y offsets.
    Loads the DEMS present in "gironde_test_data" test root
    data directory and makes the DEM translation for
    different pixel values and tests the resulting
    transform.
    """
    # When translate_dem, the values at position 0 and 3 of the
    # transform are modified
    # by the offset coordinates x and y
    # Original dem transform 0 and 3 position values before translate_dem :
    # trans[0] = 600255
    # trans[3] = 5099745

    dem, _ = load_gironde_dem
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


@pytest.mark.unit_tests
def test_translate_dem_neg_x_neg_y(load_gironde_dem):
    """
    Test the translate_dem function with negative x
    and negative y offsets.
    Loads the DEMS present in "gironde_test_data" test root
    data directory and makes the DEM translation for
    different pixel values and tests the resulting
    transform.
    """
    dem, _ = load_gironde_dem
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


@pytest.mark.unit_tests
def test_translate_dem_neg_x_pos_y(load_gironde_dem):
    """
    Test the translate_dem function with negative x
    and positive y offsets.
    Loads the DEMS present in "gironde_test_data" test root
    data directory and makes the DEM translation for
    different pixel values and tests the resulting
    transform.
    """
    dem, _ = load_gironde_dem
    # Positive y and negative x ----------------------------
    y_off_pix = 5
    x_off_pix = -4
    # Define ground truth values
    gt_offset_coord_x = 598255
    gt_offset_coord_y = 5097245

    transformed_dem = dem_tools.translate_dem(dem, x_off_pix, y_off_pix)
    # Verify that the transform of the transformed
    # dem has the ground truth values
    np.testing.assert_allclose(
        gt_offset_coord_x,
        transformed_dem["georef_transform"].data[0],
        atol=1e-05,
    )
    np.testing.assert_allclose(
        gt_offset_coord_y,
        transformed_dem["georef_transform"].data[3],
        atol=1e-02,
    )


@pytest.mark.unit_tests
def test_translate_dem_pos_x_pos_y(load_gironde_dem):
    """
    Test the translate_dem function with positive x
    and positive y offsets.
    Loads the DEMS present in "gironde_test_data" test root
    data directory and makes the DEM translation for
    different pixel values and tests the resulting
    transform.
    """
    dem, _ = load_gironde_dem
    # Positive y and positive x ----------------------------
    y_off_pix = 5
    x_off_pix = 4
    # Define ground truth values
    gt_offset_coord_x = 602255
    gt_offset_coord_y = 5097245

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
        atol=1e-05,
    )


@pytest.mark.unit_tests
def test_translate_dem_no_offset(load_gironde_dem):
    """
    Test the translate_dem function without input offset.
    Loads the DEMS present in "gironde_test_data" test root
    data directory and makes the DEM translation for
    different pixel values and tests the resulting
    transform.
    """
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
def test_reproject_dems_sampling_sec(load_gironde_dem):
    """
    Test the reproject_dems function with sampling source dem
    Loads the DEMS present in "gironde_test_data" test root
    data directory and reprojects them to test the
    obtained reprojected DEMs.

    """
    sec_orig, ref_orig = load_gironde_dem
    # Reproject dems with sampling source sec  -------------------------------
    (reproj_sec, reproj_ref, adapting_factor,) = dem_tools.reproject_dems(
        sec_orig,
        ref_orig,
        sampling_source=dem_tools.SamplingSourceParameter.SEC.value,
    )

    # Define ground truth values
    gt_intersection_roi = (600255.0, 4990745.0, 689753.076, 5090117.757)
    # Since sampling value is "sec",
    # output resolutions is "sec"'s resolution
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
        reproj_sec["image"].shape, reproj_ref["image"].shape
    )
    np.testing.assert_allclose(
        reproj_sec["image"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["image"].shape, gt_output_shape, rtol=1e-03
    )
    # Tests dems resolution
    np.testing.assert_allclose(reproj_sec.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_sec.xres, gt_output_xres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.xres, gt_output_xres, rtol=1e-02)
    # Tests dems bounds
    np.testing.assert_allclose(
        reproj_sec.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    # Test adapting_factor
    np.testing.assert_allclose(adapting_factor, gt_adapting_factor, rtol=1e-02)
    # Test dems transform
    np.testing.assert_allclose(
        reproj_sec.georef_transform, gt_output_trans, atol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.georef_transform, gt_output_trans, atol=1e-02
    )


@pytest.mark.unit_tests
def test_reproject_dems_sampling_ref(load_gironde_dem):
    """
    Test the reproject_dems function with sampling_source ref
    Loads the DEMS present in "gironde_test_data" test root
    data directory and reprojects them to test the
    obtained reprojected DEMs.

    """
    sec_orig, ref_orig = load_gironde_dem
    # Reproject dems with sampling source ref --------------------------------

    (reproj_sec, reproj_ref, adapting_factor,) = dem_tools.reproject_dems(
        sec_orig,
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
        reproj_sec["image"].shape, reproj_ref["image"].shape
    )
    np.testing.assert_allclose(
        reproj_sec["image"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["image"].shape, gt_output_shape, rtol=1e-03
    )
    # Tests dems resolution
    np.testing.assert_allclose(reproj_sec.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_sec.xres, gt_output_xres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.xres, gt_output_xres, rtol=1e-02)
    # Tests dems bounds
    np.testing.assert_allclose(
        reproj_sec.attrs["bounds"], gt_intersection_roi, rtol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.attrs["bounds"], gt_intersection_roi, rtol=1e-02
    )
    # Test adapting_factor
    np.testing.assert_allclose(adapting_factor, gt_adapting_factor, rtol=1e-02)
    # Test dems transform
    np.testing.assert_allclose(
        reproj_sec.georef_transform, gt_output_trans, rtol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.georef_transform, gt_output_trans, rtol=1e-02
    )


@pytest.mark.unit_tests
def test_reproject_dems_sampling_sec_initial_disparity(load_gironde_dem):
    """
    Test the reproject_dems function with sampling_source sec
    and initial disparity
    Loads the DEMS present in "gironde_test_data" test root
    data directory and reprojects them to test the
    obtained reprojected DEMs.

    """
    sec_orig, ref_orig = load_gironde_dem

    # Reproject dems with sampling value sec and initial disparity -------------

    (reproj_sec, reproj_ref, adapting_factor,) = dem_tools.reproject_dems(
        sec_orig,
        ref_orig,
        sampling_source=dem_tools.SamplingSourceParameter.SEC.value,
        initial_shift_x=2,
        initial_shift_y=-3,
    )

    # Define ground truth values
    gt_intersection_roi = (601255.0, 4992245.0, 690753.076977, 5091617.757489)
    # Since sampling value is "sec",
    # output resolutions is "sec"'s resolution
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
        reproj_sec["image"].shape, reproj_ref["image"].shape
    )
    np.testing.assert_allclose(
        reproj_sec["image"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["image"].shape, gt_output_shape, rtol=1e-03
    )
    # Tests dems resolution
    np.testing.assert_allclose(reproj_sec.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_sec.xres, gt_output_xres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.xres, gt_output_xres, rtol=1e-02)
    # Tests dems bounds
    np.testing.assert_allclose(
        reproj_sec.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    # Test adapting_factor
    np.testing.assert_allclose(adapting_factor, gt_adapting_factor, rtol=1e-02)
    # Test dems transform
    np.testing.assert_allclose(
        reproj_sec.georef_transform, gt_output_trans, rtol=1e-02
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
    nodata = -3
    # Create dataset from the gironde_test_data
    # DSM with the defined data, bounds,
    # transform and nodata values
    dataset = dem_tools.create_dem(
        data=data,
        img_crs=rasterio.crs.CRS.from_epsg(32630),
        transform=trans,
        input_img=cfg["input_sec"]["path"],
        bounds=bounds_dem,
        nodata=nodata,
    )
    # Define the ground truth values
    # The created dataset should have the gironde_test_data DSM georef
    gt_img_crs = "EPSG:32630"
    gt_zunit = "m"
    gt_nodata = -3
    gt_plani_unit = "m"

    # Test that the created dataset has the ground truth values
    np.testing.assert_allclose(trans, dataset.georef_transform, rtol=1e-02)
    np.testing.assert_allclose(gt_nodata, dataset.attrs["nodata"], rtol=1e-02)
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
        input_img=cfg["input_sec"]["path"],
        bounds=bounds_dem,
        nodata=nodata,
        geoid_georef=True,
    )

    # Test that the output_dataset_with_offset has
    # the same values as the gt_dataset_with_offset
    np.testing.assert_allclose(
        output_dataset_with_offset["image"].data,
        gt_data_with_offset,
        rtol=1e-02,
    )


@pytest.mark.unit_tests
def test_create_dem_with_classification_layers_dictionary():
    """
    Test the _create_dem function with input classification layers
    Creates a dem with random input data and input classification layers
    as a dictionary.
    """

    # Define data
    data = np.ones((1000, 1000))

    # Test with input classification layer as xr.DataArray ---------------
    # Initialize the data of the classification layers
    classif_data = np.full((data.shape[0], data.shape[1], 2), np.nan)
    classif_data[:, :, 0] = np.ones((data.shape[0], data.shape[1]))
    classif_data[:, :, 1] = np.ones((data.shape[0], data.shape[1])) * 2
    classif_name = ["test_first_classif", "test_second_classif"]

    # Initialize the coordinates of the classification layers
    coords_classification_layers = [
        np.arange(data.shape[0]),
        np.arange(data.shape[1]),
        classif_name,
    ]
    # Create the classif layer xr.DataArray
    seg_classif_layer = xr.DataArray(
        data=classif_data,
        coords=coords_classification_layers,
        dims=["row", "col", "indicator"],
    )
    # Create the sec dataset
    dataset_dem = dem_tools.create_dem(
        data=data, classification_layer_masks=seg_classif_layer
    )

    # Test that the classification layers have been correctly loaded
    np.testing.assert_array_equal(
        dataset_dem.classification_layer_masks.data, classif_data
    )
    np.testing.assert_array_equal(
        dataset_dem.classification_layer_masks.indicator.data, classif_name
    )


@pytest.mark.unit_tests
def test_create_dem_with_classification_layers_dataarray():
    """
    Test the _create_dem function with input classification layers
    Creates a dem with random input data and input classification layers
    as xr.DataArray
    """

    # Define data
    data = np.ones((1000, 1000))

    # Test with input classification layer as xr.DataArray ---------------
    # Initialize the data of the classification layers
    classif_data = np.full((data.shape[0], data.shape[1], 2), np.nan)
    classif_data[:, :, 0] = np.ones((data.shape[0], data.shape[1]))
    classif_data[:, :, 1] = np.ones((data.shape[0], data.shape[1])) * 2
    classif_name = ["test_first_classif", "test_second_classif"]

    # Test with input classification layer as a dictionary ---------------

    # Initialize the classification layer data as a dictionary
    classif_layer_dict: Dict = {}
    classif_layer_dict["map_arrays"] = classif_data
    classif_layer_dict["names"] = classif_name

    # Create the sec dataset
    dataset_dem_from_dict = dem_tools.create_dem(
        data=data, classification_layer_masks=classif_layer_dict
    )

    # Test that the classification layers have been correctly loaded
    np.testing.assert_array_equal(
        dataset_dem_from_dict.classification_layer_masks.data, classif_data
    )
    np.testing.assert_array_equal(
        dataset_dem_from_dict.classification_layer_masks.indicator.data,
        classif_name,
    )


@pytest.mark.unit_tests
def test_compute_waveform():
    """
    Test the compute_waveform function.
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
    Test the compute_dem_slope function
    - Loads the data present in the test root data directory
    - Creates a dem with a created array and the input data
      georeference and transform
    - Manually computes the dem slope
    - Tests that the computed slope by the function
      compute_dem_slope is the same as ground truth
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "srtm_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    from_dataset = dem_tools.load_dem(cfg["input_sec"]["path"])

    # Generate dsm with the following data and
    # "gironde_test_data" DSM's georef and resolution
    data = np.array([[1, 0, 1], [1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    dem_dataset = dem_tools.create_dem(
        data=data,
        transform=from_dataset.georef_transform.data,
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


def test_verify_fusion_layers_sec(initialize_dems_to_fuse):
    """
    Test the verify_fusion_layers function with a sec fusion
    - Loads the data present in the test root data directory
    - Manually computes different classification layers configuration
      that include fusion layers
    """
    ref, sec = initialize_dems_to_fuse

    # Test with sec fusion ---------------------------------
    # Initialize stats input configuration
    input_classif_cfg = {
        "Status": {
            "type": "segmentation",
            "classes": {
                "valid": [0],
                "KO": [1],
                "Land": [2],
                "NoData": [3],
                "Outside_detector": [4],
            },
        },
        "Slope0": {
            "type": "slope",
            "ranges": [0, 10, 25, 50, 90],
        },
        "Fusion0": {"sec": ["Slope0", "Status"], "type": "fusion"},
    }

    dem_tools.verify_fusion_layers(sec, input_classif_cfg, "sec")
    dem_tools.verify_fusion_layers(ref, input_classif_cfg, "ref")


def test_verify_fusion_layers_error_ref(initialize_dems_to_fuse):
    """
    Test the verify_fusion_layers function with a ref fusion
    - Loads the data present in the test root data directory
    - Manually computes different classification layers configuration
      that include fusion layers
    - Tests that the verify_fusion_layers raises an error when
      the layers needed to be fusioned are not present on the input dem
    """
    ref, sec = initialize_dems_to_fuse

    # Test with ref fusion ---------------------------------
    # It should not work as no ref Status exists
    # Initialize stats input configuration
    input_classif_cfg = {
        "Status": {
            "type": "segmentation",
            "classes": {
                "valid": [0],
                "KO": [1],
                "Land": [2],
                "NoData": [3],
                "Outside_detector": [4],
            },
        },
        "Slope0": {
            "type": "slope",
            "ranges": [0, 10, 25, 50, 90],
        },
        "Fusion0": {"ref": ["Slope0", "Status"], "type": "fusion"},
    }

    dem_tools.verify_fusion_layers(sec, input_classif_cfg, "sec")
    # Test that an error is raised
    with pytest.raises(ValueError):
        dem_tools.verify_fusion_layers(ref, input_classif_cfg, "ref")


def test_verify_fusion_layers_cfg_error(initialize_dems_to_fuse):
    """
    Test the verify_fusion_layers function with an error
    - Loads the data present in the test root data directory
    - Manually computes different classification layers configuration
      that include fusion layers
    - Tests that the verify_fusion_layers raises an error when
      the layers needed to be fusioned are not present on the input cfg
    """
    ref, _ = initialize_dems_to_fuse

    # Test without defining the Status layer
    # on the input cfg ---------------------------------
    # It should not work as no ref Status exists
    # Initialize stats input configuration
    input_classif_cfg = {
        "Slope0": {
            "type": "slope",
            "ranges": [0, 10, 25, 50, 90],
        },
        "Fusion0": {"ref": ["Slope0", "Status"], "type": "fusion"},
    }

    # Test that an error is raised
    with pytest.raises(ValueError):
        dem_tools.verify_fusion_layers(ref, input_classif_cfg, "ref")


def test_load_and_reproject_different_z_units():
    """
    Test that two dems loaded with different
    alti units (ie. one in cm and another in m)
    are correctly loaded and reprojected.
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Create DEM with cm zunit
    cfg["input_sec"]["path"] = os.path.join(test_data_path, "input/dem_cm.tif")
    cfg["input_sec"]["zunit"] = "cm"

    # Load original dems
    ref_orig = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec_orig = dem_tools.load_dem(cfg["input_sec"]["path"])

    reproj_sec, reproj_ref, adapting_factor = dem_tools.reproject_dems(
        sec_orig, ref_orig
    )

    # Define ground truth values
    gt_intersection_roi = (600255.0, 4990745.0, 689753.076, 5090117.757)
    # Since sampling value is "sec",
    # output resolutions is "sec"'s resolution
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
        reproj_sec["image"].shape, reproj_ref["image"].shape
    )
    np.testing.assert_allclose(
        reproj_sec["image"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["image"].shape, gt_output_shape, rtol=1e-03
    )
    # Tests dems resolution
    np.testing.assert_allclose(reproj_sec.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_sec.xres, gt_output_xres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.xres, gt_output_xres, rtol=1e-02)
    # Tests dems bounds
    np.testing.assert_allclose(
        reproj_sec.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    # Test adapting_factor
    np.testing.assert_allclose(adapting_factor, gt_adapting_factor, rtol=1e-02)
    # Test dems transform
    np.testing.assert_allclose(
        reproj_sec.georef_transform, gt_output_trans, atol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.georef_transform, gt_output_trans, atol=1e-02
    )


def test_classification_layer_mask_with_wrong_size():
    """
    Test that demcompare's load_dems function raises an error
    when the input classification layer mask
    of the dem has a different size than its support dem
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Set output_dir correctly
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        mkdir_p(tmp_dir)

        cfg["input_sec"]["classification_layers"]["Status"][
            "map_path"
        ] = os.path.join(
            test_data_path,
            "input/Small_FinalWaveBathymetry_T30TXR_20200622T105631_Status.TIF",
        )
        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, cfg)

        with pytest.raises(SystemExit):
            # Compute initialization with wrong masks
            _ = compute_initialization(tmp_cfg_file)


def test_reproject_dems_without_intersection():
    """
    Test that demcompare's reproject_dems function
    raises an error when the input dems
    do not have a common intersection.
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    cfg["input_sec"]["path"] = os.path.join(
        test_data_path, "input/reduced_Gironde.tif"
    )

    # get data srtm
    test_data_srtm_path = demcompare_test_data_path("srtm_test_data")
    cfg["input_ref"]["path"] = os.path.join(
        test_data_srtm_path, "input/srtm_ref.tif"
    )

    # Load original dems
    ref_orig = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec_orig = dem_tools.load_dem(cfg["input_sec"]["path"])

    with pytest.raises(NameError) as error_info:
        _, _, _ = dem_tools.reproject_dems(sec_orig, ref_orig)
        assert error_info.value == "ERROR: ROIs do not intersect"


def test_translate_dem_original_dem():
    """
    Test that the dem given to the
    translate function does not have its
    georeference_transform modified,
    only the returned dem does.
    """

    # Get "srtm_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("srtm_test_data")
    # Load "srtm_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    from_dataset = dem_tools.load_dem(cfg["input_sec"]["path"])

    # Define data
    data = np.array(
        [[1, 1, 1], [1, 1, 1], [-1, -32768, 1], [1, 2, -32768], [1, 1, -32768]],
        dtype=np.float32,
    )
    # Create dataset from "srtm_test_data" DSM and specific nodata value
    dem = dem_tools.create_dem(
        data=data,
        transform=from_dataset.georef_transform.data,
        img_crs=from_dataset.crs,
        nodata=-32768,
    )
    # Define pixel offsets
    y_off_pix = -5
    x_off_pix = 4

    # Translate DEM
    _ = dem_tools.translate_dem(dem, x_off_pix, y_off_pix)

    # Test that the input dem in translate_dem
    # doesn't have its georeference_transform modified
    np.testing.assert_allclose(
        from_dataset.georef_transform.data,
        dem.georef_transform,
        rtol=1e-02,
    )


def test_wrong_classification_map():
    """
    Test that the load_dem function
    raises an error when given a
    classification layer map path
    that has different dimensions
    than its support.
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Get different size classification layer
    classif_layer_path = os.path.join(
        test_data_path,
        "input/Small_FinalWaveBathymetry_T30TXR_20200622T105631_Status.TIF",
    )
    cfg["input_sec"]["classification_layers"] = classif_layer_path

    # Initialize stats input configuration
    input_classif_cfg = {
        "Status": {
            "type": "segmentation",
            "classes": {
                "valid": [0],
                "KO": [1],
                "Land": [2],
                "NoData": [3],
                "Outside_detector": [4],
            },
        }
    }

    # Test that an error is raised
    with pytest.raises(KeyError):
        _ = dem_tools.load_dem(
            cfg["input_ref"]["path"], classification_layers=input_classif_cfg
        )


def test_load_dem_with_roi_image_coords():
    """
    Test that coords image ROI are correctly
    returned by the load_input_dems function
    Input data:
    - input DEMs present in "gironde_test_data" test root data directory
    Validation data:
    - ROI characteristics obtained by rasterio
    Validation process:
    - Load the dem with the load_dem function.
    - Verify that the loaded dem has a correct ROI
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # add roi to cfg
    cfg["input_ref"]["roi"] = {"x": 0, "y": 0, "w": 500, "h": 500}
    ref, _ = load_input_dems(cfg)

    # Instantiate ground truth ROI thanks to rasterio
    src_dem = rasterio.open(cfg["input_ref"]["path"])
    window_dem = rasterio.windows.Window(
        0,
        0,
        500,
        500,
    )
    left, bottom, right, top = rasterio.windows.bounds(
        window_dem, src_dem.transform
    )
    gt_roi = rasterio.coords.BoundingBox(left, bottom, right, top)

    assert gt_roi == ref.attrs["bounds"]


def test_load_dem_with_roi_ground_coords():
    """
    Test that grounds coords image ROI are correctly
    returned by the load_input_dems function
    Input data:
    - input DEMs present in "strm_test_data" test root data directory
    Validation data:
    - Manually computed ROI and BoundingBox obtained by rasterio
    Validation process:
    - Load the dem with the load_dem function.
    - Verify that the loaded dem has a correct ROI
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # add roi to cfg
    cfg["input_ref"]["roi"] = {
        "left": -1.7413878040009774,
        "bottom": 45.41864925736226,
        "right": -1.2205544690009773,
        "top": 45.93948259236226,
    }
    ref, _ = load_input_dems(cfg)

    # Instantiate ground truth ROI thanks to rasterio
    gt_roi = rasterio.coords.BoundingBox(
        -1.7413878040009774,
        45.41864925736226,
        -1.2205544690009773,
        45.93948259236226,
    )

    assert gt_roi == ref.attrs["bounds"]
