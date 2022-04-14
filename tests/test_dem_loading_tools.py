#!/usr/bin/env python
# coding: utf8
# Disable the protected-access to test the functions
# pylint:disable=protected-access
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
This module contains functions to test all the methods in
dem_loading_tools module.
"""

# Standard imports
import os

# Third party imports
import numpy as np
import pytest
import rasterio

from demcompare import dem_loading_tools
from demcompare.initialization import read_config_file

# Tests helpers
from .helpers import demcompare_path, demcompare_test_data_path


@pytest.mark.unit_tests
def test_load_dem():
    """
    Test the load_dem function
    Loads the data present in "standard" test root
    data directory and tests the loaded DEM Dataset.

    """
    # Get "standard" test root data directory absolute path
    test_data_path = demcompare_test_data_path("standard")
    # Load "standard" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    dem = dem_loading_tools.load_dem(cfg["inputRef"]["path"])

    # Define ground truth values
    gt_no_data = -32768.0
    gt_xres = 0.000833333333333333
    gt_yres = -0.0008333333333333334
    gt_plani_unit = "deg"
    gt_zunit = "m"
    gt_georef = "EPSG:4326"
    gt_geoid_georef = False
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
    assert gt_geoid_georef == dem.attrs["geoid_georef"]
    assert gt_shape == dem["im"].shape
    np.testing.assert_allclose(gt_transform, dem.trans, rtol=1e-02)


@pytest.mark.unit_tests
def test_create_dataset():
    """
    Test the _create_dataset function
    Creates a dataset with the data present in
    "classification_layer" test root data directory and tests
    the obtained DEM Dataset.
    """

    # Test with "classification_layer" input dem
    # Get "classification_layer" test
    # root data directory absolute path
    test_data_path = demcompare_test_data_path("classification_layer")
    # Load "classification_layer" demcompare
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
    # Create dataset from the classification_layer
    # DSM with the defined data, bounds,
    # transform and nodata values
    dataset = dem_loading_tools.create_dataset(
        data=data,
        img_crs=rasterio.crs.CRS.from_epsg(32630),
        transform=trans,
        input_img=cfg["inputDSM"]["path"],
        bounds=bounds_dem,
        no_data=nodata,
    )
    # Define the ground truth values
    # The created dataset should have the classification_layer DSM georef
    gt_img_crs = "EPSG:32630"
    gt_geoid_georef = False
    gt_zunit = "m"
    gt_no_data = -32768
    gt_plani_unit = "m"

    # Test that the created dataset has the ground truth values
    np.testing.assert_allclose(trans, dataset.trans, rtol=1e-02)
    np.testing.assert_allclose(gt_no_data, dataset.attrs["no_data"], rtol=1e-02)
    assert gt_plani_unit == dataset.attrs["plani_unit"]
    assert gt_zunit == dataset.attrs["zunit"]
    np.testing.assert_allclose(bounds_dem, dataset.attrs["bounds"], rtol=1e-02)
    assert gt_img_crs == dataset.attrs["crs"]
    assert gt_geoid_georef == dataset.attrs["geoid_georef"]

    # Test with geoid_georef set to True ---------------

    # Define geoid path
    geoid_path = demcompare_path("geoid/egm96_15.gtx")
    # Get geoid offset of the dataset
    output_arr_offset = dem_loading_tools.get_geoid_offset(dataset, geoid_path)
    # Add the geoid offset to the dataset
    gt_data_with_offset = dataset["im"].data + output_arr_offset

    # Compute the dataset with the geoid_georef parameter set to True
    output_dataset_with_offset = dem_loading_tools.create_dataset(
        data=data,
        img_crs=rasterio.crs.CRS.from_epsg(32630),
        transform=trans,
        input_img=cfg["inputDSM"]["path"],
        bounds=bounds_dem,
        no_data=nodata,
        geoid_georef=True,
    )

    # Test that the output_dataset_with_offset has
    # the same values as the gt_dataset_with_offset
    np.testing.assert_allclose(
        output_dataset_with_offset["im"].data,
        gt_data_with_offset,
        rtol=1e-02,
    )


@pytest.mark.unit_tests
def test_compute_altitude_diff():
    """
    Test compute_altitude_diff function
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

    dem_to_align_dataset = dem_loading_tools.create_dataset(data=dem_to_align)
    ref_dataset = dem_loading_tools.create_dataset(data=ref)

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

    diff_dataset = dem_loading_tools.compute_altitude_diff(
        ref_dataset, dem_to_align_dataset
    )
    # Test that the output difference is the same as ground_truth
    np.testing.assert_array_equal(diff_gt, diff_dataset["im"].data)

    # Create input datasets
    dem_to_align = np.array(
        [[1, 1, 1], [1, 1, 1], [-1, -33, 1], [1, 2, -33], [1, 1, -33]],
        dtype=np.float32,
    )
    ref = np.array(
        [[3, 3, 3], [1, 2, 1], [-9999, 0, 1], [1, 1, 0], [1, 1, -9999]],
        dtype=np.float32,
    )

    dem_to_align_dataset = dem_loading_tools.create_dataset(
        data=dem_to_align, no_data=-33
    )
    ref_dataset = dem_loading_tools.create_dataset(data=ref)

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
    diff_dataset = dem_loading_tools.compute_altitude_diff(
        ref_dataset, dem_to_align_dataset
    )
    # Test that the output difference is the same as ground_truth
    np.testing.assert_array_equal(diff_gt, diff_dataset["im"].data)


@pytest.mark.unit_tests
def test_crop_dataset_with_roi():
    """
    Test the _crop_dataset_with_roi function
    Creates a dataset with the data present in
    "standard" test root data directory and tests
    the outputs of the _crop_dataset_with_roi function.
    """
    # Get "standard" test root data directory absolute path
    test_data_path = demcompare_test_data_path("standard")
    # Load "standard" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load src in rasterio mode
    src_static = rasterio.open(cfg["inputDSM"]["path"])

    # Define bounding_box coordinates -----------

    left = 40.5
    bottom = 38.0
    right = 44.0
    top = 41.0

    # Define ground_truth polygon
    gt_polygon = [
        [left, bottom],
        [right, bottom],
        [right, top],
        [left, top],
        [left, bottom],
    ]
    geom_like_polygon = {"type": "Polygon", "coordinates": [gt_polygon]}
    # Obtain ground truth cropped dems
    gt_cropped_dem, gt_cropped_dem_transform = rasterio.mask.mask(
        src_static, [geom_like_polygon], all_touched=True, crop=True
    )

    (
        output_cropped_dem,
        output_cropped_dem_transform,
    ) = dem_loading_tools._crop_dataset_with_roi(
        src_static, [left, bottom, right, top]
    )
    # Test that the output_cropped_dem is the same as ground_truth
    np.testing.assert_allclose(gt_cropped_dem, output_cropped_dem, rtol=1e-04)
    # Test that the output_cropped_dem_transform is the same as ground_truth
    np.testing.assert_allclose(
        gt_cropped_dem_transform, output_cropped_dem_transform, rtol=1e-04
    )

    # Define bounding_box coordinates outside of the DEM scope -----------
    left = -3.0
    bottom = 5.0
    right = 1.0
    top = 0.5

    # Test that an error is raised
    with pytest.raises(ValueError):
        dem_loading_tools._crop_dataset_with_roi(
            src_static, [left, bottom, right, top]
        )


@pytest.mark.unit_tests
def test_reproject_dems():
    """
    Test the reproject_dems function
    Loads the DEMS present in "classification_layer" test root
    data directory and reprojects them to test the
    obtained reprojected DEMs.
    """
    # Get "classification_layer" test root data directory absolute path
    test_data_path = demcompare_test_data_path("classification_layer")
    # Load "classification_layer" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load original dems
    ref_orig = dem_loading_tools.load_dem(cfg["inputRef"]["path"])
    dem_to_align_orig = dem_loading_tools.load_dem(cfg["inputDSM"]["path"])

    # Reproject dems with sampling value dem
    reproj_dem_to_align, reproj_ref = dem_loading_tools.reproject_dems(
        dem_to_align_orig, ref_orig, sampling_source="dem_to_align"
    )

    # Define ground truth values
    gt_intersection_roi = (600255.0, 4990745.0, 689753.076, 5090117.757)
    # Since sampling value is "dem_to_align",
    # output resolutions is "dem_to_align"'s resolution
    gt_output_yres = -500.00
    gt_output_xres = 500.00
    gt_output_shape = (199, 179)

    # Test that both the output dems have the same gt values
    # Tests dems shape
    np.testing.assert_array_equal(
        reproj_dem_to_align["im"].shape, reproj_ref["im"].shape
    )
    np.testing.assert_allclose(
        reproj_dem_to_align["im"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["im"].shape, gt_output_shape, rtol=1e-03
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

    # Reproject dems with sampling value ref
    reproj_dem_to_align, reproj_ref = dem_loading_tools.reproject_dems(
        dem_to_align_orig, ref_orig, sampling_source="ref"
    )

    # Define ground truth values
    gt_intersection_roi = (-1.726, 45.039, -0.602, 45.939)
    # Since sampling value is "ref", output resolutions is "ref"'s resolution
    gt_output_yres = -0.0010416
    gt_output_xres = 0.0010416
    gt_output_shape = (865, 1079)

    # Test that both the output dems have the same gt values
    # Tests dems shape
    np.testing.assert_array_equal(
        reproj_dem_to_align["im"].shape, reproj_ref["im"].shape
    )
    np.testing.assert_allclose(
        reproj_dem_to_align["im"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["im"].shape, gt_output_shape, rtol=1e-03
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


@pytest.mark.unit_tests
def test_create_dataset_from_dataset():
    """
    Test create_dataset_from_dataset function
    Creates a dataset from an input array and an input
    dataset to test the obtained output.
    """

    # Test without input dataset  -----------------------------
    # Define data
    data = np.array(
        [[1, 1, 1], [1, 1, 1], [-1, -9999, 1], [1, 2, -9999], [1, 1, -9999]],
        dtype=np.float32,
    )
    output_dataset = dem_loading_tools.create_dataset_from_dataset(
        img_array=data
    )
    # Define ground truth values
    # Since no dataset was given, the output
    # dataset should have the default values
    gt_xres = 1
    gt_yres = 1
    gt_nodata = -9999
    gt_data = np.array(
        [[1, 1, 1], [1, 1, 1], [-1, np.nan, 1], [1, 2, np.nan], [1, 1, np.nan]],
        dtype=np.float32,
    )
    # Test that the output_dataset has the gt_values
    assert gt_xres == output_dataset.attrs["xres"]
    assert gt_yres == output_dataset.attrs["yres"]
    assert gt_nodata == output_dataset.attrs["no_data"]
    np.testing.assert_array_equal(gt_data, output_dataset["im"].data)

    # Test with input dataset  -----------------------------

    # Get "standard" test root data directory absolute path
    test_data_path = demcompare_test_data_path("standard")
    # Load "standard" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    from_dataset = dem_loading_tools.load_dem(cfg["inputDSM"]["path"])
    # Define data
    data = np.array(
        [[1, 1, 1], [1, 1, 1], [-1, -32768, 1], [1, 2, -32768], [1, 1, -32768]],
        dtype=np.float32,
    )
    # Create dataset from "standard" DSM and specific no_data value
    output_dataset = dem_loading_tools.create_dataset_from_dataset(
        img_array=data, from_dataset=from_dataset, no_data=-32768
    )
    # Define ground truth values
    # The output_dataset should have the
    # from_dataset resolution and its specified no_data
    gt_xres = from_dataset.attrs["xres"]
    gt_yres = from_dataset.attrs["yres"]
    gt_nodata = -32768
    gt_data = np.array(
        [[1, 1, 1], [1, 1, 1], [-1, np.nan, 1], [1, 2, np.nan], [1, 1, np.nan]],
        dtype=np.float32,
    )
    # Test that the output_dataset has the gt_values
    assert gt_xres == output_dataset.attrs["xres"]
    assert gt_yres == output_dataset.attrs["yres"]
    assert gt_nodata == output_dataset.attrs["no_data"]
    np.testing.assert_array_equal(gt_data, output_dataset["im"].data)
