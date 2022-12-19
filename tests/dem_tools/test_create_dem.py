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
This module contains functions to test the create_dem function.
"""
# pylint:disable=protected-access
from typing import Dict

# Third party imports
import numpy as np
import pytest
import rasterio
import xarray as xr

# Demcompare imports
from demcompare import dataset_tools, dem_tools

# Tests helpers
from tests.helpers import demcompare_path


@pytest.mark.unit_tests
def test_create_dem():
    """
    Test create_dem function
    Input data:
    - A manually created np.array as data
    - A manually created bounds, transform and nodata value
    Validation data:
    - The values used to create the dem: data, bounds_dem,
      trans, nodata
    - The ground truth attributes of the created dem:
      gt_img_crs, gt_zunit, gt_plani_unit
    Validation process:
    - Create the dem using the create_dem function
    - Check that the obtained dem's values
      are the same as the ones given to the function
    - Check that the obtained dem's attributes are the same
      as ground truth
    - Checked function : dem_tools's create_dem
    """

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
        bounds=bounds_dem,
        nodata=nodata,
    )
    # Define the ground truth values
    # The created dataset should have the gironde_test_data DSM georef
    gt_img_crs = "EPSG:32630"
    gt_zunit = "m"
    gt_plani_unit = "m"

    # Test that the created dataset has the ground truth values
    np.testing.assert_allclose(trans, dataset.georef_transform, rtol=1e-02)
    np.testing.assert_allclose(nodata, dataset.attrs["nodata"], rtol=1e-02)
    assert gt_plani_unit == dataset.attrs["plani_unit"]
    assert gt_zunit == dataset.attrs["zunit"]
    np.testing.assert_allclose(bounds_dem, dataset.attrs["bounds"], rtol=1e-02)
    assert gt_img_crs == dataset.attrs["crs"]


@pytest.mark.unit_tests
def test_create_dem_with_geoid_georef():
    """
    Test create_dem function with geoid georef
    Input data:
    - A manually created np.array as data
    - A manually created bounds, transform and nodata value
    - The geoid present in the geoid/egm96_15.gtx directory
    Validation data:
    - The values used to create the dem: bounds_dem,
      trans, nodata
    - The ground truth attributes of the created dem:
      gt_img_crs, gt_zunit, gt_plani_unit
    - The value of the dem's image when its geoid offset
      has been applied
    Validation process:
    - Create the dem using the create_dem function
    - Compute the dem's geoid offset using the function
      _get_geoid_offset
    - Check that the obtained dem's values
      are the same as the ones given to the function
    - Check that the obtained dem's attributes are the same
      as ground truth
    - Checked function : dem_tools's create_dem
    """

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
        bounds=bounds_dem,
        nodata=nodata,
    )

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
    Test create_dem function with input classification_layer_masks
    as a dictionnary
    Input data:
    - A manually created np.array as data
    - A manually created classification_layer_masks dictionnary
      containing two masks called "test_first_classif", "test_second_classif"
    Validation data:
    - The geoid present in the geoid/egm96_15.gtx directory
    Validation data:
    - The classification_layer_masks dictionnary used to create the dem
    Validation process:
    - Create the dem using the create_dem function with the input
      data and the classification_layer_masks dictionnary
    - Check that the obtained dem contains the classification layer masks
      information
    - Checked function : dem_tools's create_dem
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
    Test create_dem function with input classification_layer_masks
    as an xr.DataArray
    Input data:
    - A manually created np.array as data
    - A manually created classification_layer_masks as an xr.DataArray
      containing two masks called "test_first_classif", "test_second_classif"
    Validation data:
    - The geoid present in the geoid/egm96_15.gtx directory
    Validation data:
    - The classification_layer_masks xr.DataArray used to create the dem
    Validation process:
    - Create the dem using the create_dem function with the input
      data and the classification_layer_masks as an xr.DataArray
    - Check that the obtained dem contains the classification layer masks
      information
    - Checked function : dem_tools's create_dem
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
