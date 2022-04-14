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
dem_projection_tools module.
"""

# Standard imports
import os

# Third party imports
import numpy as np
import pytest
import rasterio
import scipy

from demcompare import dem_loading_tools, dem_projection_tools
from demcompare.initialization import read_config_file

# Tests helpers
from .helpers import demcompare_path, demcompare_test_data_path


@pytest.mark.unit_tests
def test_pix_to_coord():
    """
    Test _pix_to_coord function
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
    x_coord, y_coord = dem_projection_tools._pix_to_coord(trans, y_pix, x_pix)
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
    x_coord, y_coord = dem_projection_tools._pix_to_coord(trans, y_pix, x_pix)
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
    x_coord, y_coord = dem_projection_tools._pix_to_coord(trans, y_pix, x_pix)
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
    x_coord, y_coord = dem_projection_tools._pix_to_coord(trans, y_pix, x_pix)
    # Define ground truth coords
    x_coord_gt = 594080
    y_coord_gt = 5097195

    np.testing.assert_allclose(x_coord_gt, x_coord, rtol=1e-03)
    np.testing.assert_allclose(y_coord_gt, y_coord_gt, rtol=1e-03)


@pytest.mark.unit_tests
def test_translate_dataset():
    """
    Test the _translate_dataset function
    Loads the DEMS present in "classification_layer" test root
    data directory and makes the DEM translation for
    different pixel values and tests the resulting
    transform.
    """
    # Test with "classification_layer" test root input DEMs
    # and sampling value ref
    # Get "classification_layer" test root data directory absolute path
    test_data_path = demcompare_test_data_path("classification_layer")
    # Load "classification_layer" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    dem = dem_loading_tools.load_dem(cfg["inputDSM"]["path"])

    # When translate_dataset, the values at position 0 and 3 of the
    # transform are modified
    # by the offset coordinates x and y
    # Original dem transform 0 and 3 position values before translate_dataset :
    # trans[0] = 600255
    # trans[3] = 5099745

    # Negative y and positive x -----------------------------
    y_off_pix = -5
    x_off_pix = 4
    # Define ground truth values
    gt_offset_coord_x = 602255
    gt_offset_coord_y = 5102245

    transformed_dem = dem_projection_tools.translate_dataset(
        dem, x_off_pix, y_off_pix
    )
    # Verify that the transform of the transformed
    # dem has the ground truth values
    np.testing.assert_allclose(
        gt_offset_coord_x, transformed_dem["trans"].data[0], rtol=1e-02
    )
    np.testing.assert_allclose(
        gt_offset_coord_y, transformed_dem["trans"].data[3], rtol=1e-02
    )

    # Negative y and negative x --------------------------
    y_off_pix = -5
    x_off_pix = -4
    # Define ground truth values
    gt_offset_coord_x = 600255
    gt_offset_coord_y = 5104745

    transformed_dem = dem_projection_tools.translate_dataset(
        dem, x_off_pix, y_off_pix
    )
    # Verify that the transform of the transformed
    # dem has the ground truth values
    np.testing.assert_allclose(
        gt_offset_coord_x, transformed_dem["trans"].data[0], rtol=1e-02
    )
    np.testing.assert_allclose(
        gt_offset_coord_y, transformed_dem["trans"].data[3], rtol=1e-02
    )

    # Positive y and negative x ----------------------------
    y_off_pix = 5
    x_off_pix = -4
    # Define ground truth values
    gt_offset_coord_x = 598255
    gt_offset_coord_y = 5102245

    transformed_dem = dem_projection_tools.translate_dataset(
        dem, x_off_pix, y_off_pix
    )
    # Verify that the transform of the transformed
    # dem has the ground truth values
    np.testing.assert_allclose(
        gt_offset_coord_x, transformed_dem["trans"].data[0], rtol=1e-02
    )
    np.testing.assert_allclose(
        gt_offset_coord_y, transformed_dem["trans"].data[3], rtol=1e-02
    )

    # Positive y and positive x ----------------------------
    y_off_pix = 5
    x_off_pix = -4
    # Define ground truth values
    gt_offset_coord_x = 596255
    gt_offset_coord_y = 5099745

    transformed_dem = dem_projection_tools.translate_dataset(
        dem, x_off_pix, y_off_pix
    )
    # Verify that the transform of the transformed
    # dem has the ground truth values
    np.testing.assert_allclose(
        gt_offset_coord_x, transformed_dem["trans"].data[0], rtol=1e-02
    )
    np.testing.assert_allclose(
        gt_offset_coord_y, transformed_dem["trans"].data[3], rtol=1e-02
    )

    # No offset ------------------------------------
    y_off_pix = 0
    x_off_pix = 0
    # Define ground truth values
    gt_offset_coord_x = 600255
    gt_offset_coord_y = 5099745

    transformed_dem = dem_projection_tools.translate_dataset(
        dem, x_off_pix, y_off_pix
    )
    # Verify that the transform of the transformed
    # dem has the ground truth values
    np.testing.assert_allclose(
        gt_offset_coord_x, transformed_dem["trans"].data[0], rtol=1e-02
    )
    np.testing.assert_allclose(
        gt_offset_coord_y, transformed_dem["trans"].data[3], rtol=1e-02
    )


@pytest.mark.unit_tests
def test_compute_offset_bounds():
    """
    Test the compute_offset_bounds function
    Loads the DEMS present in "standard" and "classification_layer"
    test root data directory and computes the coordinate offset
    bounds for a given pixellic offset to test the resulting
    bounds.
    """

    # Test with "standard" input dem
    # Get "classification_layer" test
    # root data directory absolute path
    test_data_path = demcompare_test_data_path("standard")
    # Load "classification_layer" demcompare
    # config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    dem = dem_loading_tools.load_dem(cfg["inputDSM"]["path"])

    # Pixellic offset
    dy_px = 0.00417
    dx_px = 0.0025

    # Define the ground truth reprojected values
    gt_ulx = 40.00000
    gt_uly = 40.00000
    gt_lrx = 40.83083
    gt_lry = 39.17083

    ulx, uly, lrx, lry = dem_projection_tools.compute_offset_bounds(
        -dy_px, dx_px, dem["im"].shape, dem["trans"].data
    )
    # Test that the reprojected offsets are the same as ground_truth
    np.testing.assert_allclose(ulx, gt_ulx, rtol=1e-04)
    np.testing.assert_allclose(uly, gt_uly, rtol=1e-04)
    np.testing.assert_allclose(lrx, gt_lrx, rtol=1e-04)
    np.testing.assert_allclose(lry, gt_lry, rtol=1e-04)

    # Test with "classification_layer" input dem
    # Get "classification_layer" test
    # root data directory absolute path
    test_data_path = demcompare_test_data_path("classification_layer")
    # Load "classification_layer" demcompare
    # config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    dem = dem_loading_tools.load_dem(cfg["inputDSM"]["path"])

    # Pixellic offset
    dy_nuth = 0.41903
    dx_nuth = -1.43664

    # Define the ground truth reprojected values
    gt_ulx = 599536.68
    gt_uly = 5099954.515
    gt_lrx = 708536.68
    gt_lry = 4990954.515

    ulx, uly, lrx, lry = dem_projection_tools.compute_offset_bounds(
        -dy_nuth, dx_nuth, dem["im"].shape, dem["trans"].data
    )

    # Test that the reprojected offsets are the same as ground_truth
    np.testing.assert_allclose(ulx, gt_ulx, rtol=1e-04)
    np.testing.assert_allclose(uly, gt_uly, rtol=1e-04)
    np.testing.assert_allclose(lrx, gt_lrx, rtol=1e-04)
    np.testing.assert_allclose(lry, gt_lry, rtol=1e-04)


@pytest.mark.unit_tests
def test_get_slope():
    """
    Test the get_slope function
    Loads the DEMS present in "classification_layer"
    test root data directory and computes the DEM slope
    to test the resulting values.
    """

    # Get "classification_layer" test root data directory absolute path
    test_data_path = demcompare_test_data_path("classification_layer")
    # Load "standard" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    from_dataset = dem_loading_tools.load_dem(cfg["inputDSM"]["path"])

    # Generate dsm with the following data and
    # "classification_layer" DSM's georef and resolution
    data = np.array([[1, 0, 1], [1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    dem_dataset = dem_loading_tools.create_dataset(
        data=data,
        transform=from_dataset.trans.data,
        img_crs=rasterio.crs.CRS.from_epsg(32630),
    )

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

    output_slope = dem_projection_tools.get_slope(dem_dataset)
    # Test that the output_slope is the same as ground_truth
    np.testing.assert_allclose(gt_slope, output_slope, rtol=1e-02)


@pytest.mark.unit_tests
def test_reproject_dataset():
    """
    Test the reproject_dataset function
    Loads the DEMS present in "standard" and "classification_layer"
    test root data directory and reprojects one
    onto another to test the obtained
    reprojected DEMs.
    """
    # Generate the "reproject_on_dataset" with the
    # following data, transform and nodata
    # and "standard" DSM's georef
    data = np.ones((1000, 1000))
    trans = np.array(
        [
            4.00000000e01,
            8.33333333e-04,
            0.00000000e00,
            4.00000000e01,
            0.00000000e00,
            -8.33333333e-04,
        ]
    )
    nodata = -33

    # Get "standard" test
    # root data directory absolute path
    test_data_path = demcompare_test_data_path("standard")
    # Load "classification_layer" demcompare
    # config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    reproject_on_dataset = dem_loading_tools.create_dataset(
        data=data,
        transform=trans,
        input_img=cfg["inputDSM"]["path"],
        no_data=nodata,
    )

    # Generate the "dataset_to_be_reprojected" with
    # the following data, transform and nodata
    # and "classification_layer" DSM's georef
    data = np.ones((1000, 1000))
    trans = np.array([600000, 50, 0, 600000, 0, 50])
    nodata = -32768

    # Get "classification_layer" test
    # root data directory absolute path
    test_data_path = demcompare_test_data_path("classification_layer")
    # Load "classification_layer" demcompare
    # config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    dataset_to_be_reprojected = dem_loading_tools.create_dataset(
        data=data,
        transform=trans,
        input_img=cfg["inputDSM"]["path"],
        no_data=nodata,
    )

    # Reproject the dataset_to_be_reprojected on
    # reproject_on_dataset
    output_reprojected_dataset = dem_projection_tools.reproject_dataset(
        dataset_to_be_reprojected, reproject_on_dataset
    )
    # Test that the output dataset now has the
    # transform of reproject_on_dataset
    np.testing.assert_allclose(
        reproject_on_dataset.trans, output_reprojected_dataset.trans, rtol=1e-02
    )
    # Test that the output dataset now has the
    # georef of reproject_on_dataset
    assert (
        reproject_on_dataset.attrs["crs"]
        == output_reprojected_dataset.attrs["crs"]
    )
    # Test that the output dataset still has
    # its original no_data value
    np.testing.assert_allclose(
        dataset_to_be_reprojected.attrs["no_data"],
        output_reprojected_dataset.attrs["no_data"],
        rtol=1e-02,
    )


@pytest.mark.unit_tests
def test_interpolate_geoid():
    """
    Test the _interpolate_geoid function
    Interpolates the default egm96_15 geoid
    with a given latitude and longitude input values
    to test the obtained geoid offsets.
    """
    # Load geoid
    geoid_data_path = demcompare_path("geoid/egm96_15.gtx")
    geoid_dataset = rasterio.open(geoid_data_path)

    # Define geoid parameters and values
    step_lon = 0.25
    step_lat = 0.25
    origin_lon = -180
    last_lon = 180
    origin_lat = -90
    last_lat = 90.25
    geoid_values = geoid_dataset.read(1)[::-1, :].transpose()
    # Obtain geoid grid
    lon = np.arange(origin_lon, last_lon, step_lon)
    lat = np.arange(origin_lat, last_lat, step_lat)
    geoid_grid_coordinates = (lon, lat)

    # Define coords to be interpolated within the geoid scope ----------------
    coords = np.array([[-0.07641488, 51.52168219]])

    # Obtain ground_truth interpolated coordinates
    gt_interp_geoid = scipy.interpolate.interpn(
        geoid_grid_coordinates,
        geoid_values,
        coords,
        method="linear",
        bounds_error=True,
        fill_value=None,
    )

    output_interp_geoid = dem_projection_tools._interpolate_geoid(
        geoid_data_path, coords
    )
    # Test that the output_interp_geoid is the same as ground_truth
    np.testing.assert_allclose(gt_interp_geoid, output_interp_geoid, rtol=1e-04)

    # Define coords to be interpolated outside the scope ----------------

    coords = np.array([[-181, 51.52168219]])

    # Test that an error is raised
    with pytest.raises(ValueError):
        dem_projection_tools._interpolate_geoid(geoid_data_path, coords)


def test_get_geoid_offset():
    """
    Test the _get_geoid_offset function
    Loads the DEMS present in "standard" test root data
    directory and projects it on the geoid to test
    the obtained dataset's geoid offset values.
    """
    # Get "standard" test root data directory absolute path
    test_data_path = demcompare_test_data_path("standard")
    # Load "standard" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Geoid path
    geoid_path = demcompare_path("geoid/egm96_15.gtx")

    # Define data
    data = np.ones((2, 2))

    # Define transformation --------------------
    trans = np.array(
        [
            4.00000000e01,
            8.33333333e-04,
            0.00000000e00,
            4.00000000e01,
            0.00000000e00,
            -8.33333333e-04,
        ]
    )
    nodata = -32768
    # Create dataset from the standard
    # DSM with the defined data, bounds,
    # transform and nodata values
    dataset = dem_loading_tools.create_dataset(
        data=data,
        transform=trans,
        input_img=cfg["inputDSM"]["path"],
        no_data=nodata,
    )

    # Define data coordinates
    gt_data_coords = np.array(
        [
            [40.0, 40.0],
            [40.00083333, 40.0],
            [40.0, 39.99916667],
            [40.00083333, 39.99916667],
        ]
    )

    # Get interpolated geoid values
    gt_interp_geoid = dem_projection_tools._interpolate_geoid(
        geoid_path, gt_data_coords, interpol_method="linear"
    )
    gt_arr_offset = np.reshape(gt_interp_geoid, dataset["im"].data.shape)

    # Get offset values
    output_arr_offset = dem_projection_tools.get_geoid_offset(
        dataset, geoid_path
    )

    # Test that the output_arr_offset is the same as ground_truth
    np.testing.assert_allclose(gt_arr_offset, output_arr_offset, rtol=1e-04)

    # Define transformation that will compute the data coordinates
    # outside of the geoid scope --------------------
    trans = np.array(
        [
            182.0,
            8.33333333e-04,
            0.00000000e00,
            91,
            0.00000000e00,
            -8.33333333e-04,
        ]
    )
    nodata = -32768
    # Create dataset from the classification_layer
    # DSM with the defined data, bounds,
    # transform and nodata values
    dataset = dem_loading_tools.create_dataset(
        data=data,
        transform=trans,
        input_img=cfg["inputDSM"]["path"],
        no_data=nodata,
    )

    # Test that an error is raised
    with pytest.raises(ValueError):
        # Get geoid values
        dem_projection_tools.get_geoid_offset(dataset, geoid_path)
