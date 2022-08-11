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
This module contains functions to test the
methods in the classification layer class.
"""

# Third party imports
import numpy as np
import pytest

# Demcompare imports
from demcompare import dem_tools
from demcompare.classification_layer.classification_layer import (
    ClassificationLayer,
)

_DEFAULT_TEST_METRICS = [
    "mean",
    "median",
    "max",
    "min",
    "sum",
    {"percentil_90": {"remove_outliers": "False"}},
    "squared_sum",
    "nmad",
    "rmse",
    "std",
]


@pytest.mark.unit_tests
def test_get_outliers_free_mask():
    """
    Test the _get_outliers_free_mask function
    Manually computes an input array and filters it,
    and tests that the resulting
    arrays form the _get_outliers_free_mask are the
    same.
    """

    # Generate dsm with the following data and
    # "gironde_test_data" DSM's georef and resolution
    data = np.array(
        [
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [-9999, 1, 1],
            [0, 1, 1],
            [0, 20, 1],
        ],
        dtype=np.float32,
    )
    dem_dataset = dem_tools.create_dem(
        data=data,
    )

    # Compute dem's slope
    dem_dataset = dem_tools.compute_dem_slope(dem_dataset)
    # Classification layer configuration
    layer_name = "Slope0"
    clayer = {
        "type": "slope",
        "ranges": [0, 5, 10, 25, 45],
        "save_results": False,
        "output_dir": "",
        "metrics": _DEFAULT_TEST_METRICS,
    }

    # Initialize slope classification layer object
    slope_classif_layer_ = ClassificationLayer(
        name=layer_name,
        classification_layer_kind=str(clayer["type"]),
        dem=dem_dataset,
        cfg=clayer,
    )

    output_filtered_mask = slope_classif_layer_._get_outliers_free_mask(
        data, -9999
    )

    # All nodata values of the input data
    array_without_nodata = np.array(
        [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 20, 1],
        dtype=np.float32,
    )
    mu = np.mean(array_without_nodata)
    sigma = np.std(array_without_nodata)
    upper_threshold = mu + 3 * sigma
    lower_threshold = mu - 3 * sigma

    gt_filtered_mask = np.ones(data.shape) * True
    gt_filtered_mask[np.where(data > upper_threshold)] = False
    gt_filtered_mask[np.where(data < lower_threshold)] = False

    np.testing.assert_allclose(
        gt_filtered_mask, output_filtered_mask, rtol=1e-02
    )


@pytest.mark.unit_tests
def test_get_non_mask():
    """
    Test the _get_outliers_free_mask function
    Manually computes an input array and filters it,
    and tests that the resulting
    arrays form the _get_outliers_free_mask are the
    same.
    """
    # Classification layer configuration
    layer_name = "Slope0"
    clayer = {
        "type": "slope",
        "ranges": [0, 5, 10, 25, 45],
        "save_results": False,
        "output_dir": "",
        "metrics": _DEFAULT_TEST_METRICS,
    }

    data = np.array(
        [[0, 1, 1], [0, -9999, 1], [-9999, 1, 1], [-9999, 1, 1]],
        dtype=np.float32,
    )
    dem_dataset = dem_tools.create_dem(
        data=data,
    )
    # Compute dem's slope
    dem_dataset = dem_tools.compute_dem_slope(dem_dataset)
    # Initialize slope classification layer object
    slope_classif_layer_ = ClassificationLayer(
        name=layer_name,
        classification_layer_kind=str(clayer["type"]),
        dem=dem_dataset,
        cfg=clayer,
    )

    # Test with custom nodata value -------------------------------
    # Compute no nan mask
    output_nonan_mask = slope_classif_layer_._get_nonan_mask(data, -9999)
    # Ground truth no nan mask
    gt_nonan_mask = np.array(
        [
            [True, True, True],
            [True, False, True],
            [False, True, True],
            [False, True, True],
        ],
        dtype=np.float32,
    )
    # Test that the computed no nan mask is equal to ground truth
    np.testing.assert_allclose(gt_nonan_mask, output_nonan_mask, rtol=1e-02)

    # Test with default nodata value -------------------------------
    data = np.array(
        [[0, 1, 1], [0, np.nan, np.nan], [np.nan, 1, 1], [np.nan, 1, 1]],
        dtype=np.float32,
    )
    # Compute no nan mask
    output_nonan_mask = slope_classif_layer_._get_nonan_mask(data)
    # Ground truth no nan mask
    gt_nonan_mask = np.array(
        [
            [True, True, True],
            [True, False, False],
            [False, True, True],
            [False, True, True],
        ],
        dtype=np.float32,
    )
    # Test that the computed no nan mask is equal to ground truth
    np.testing.assert_allclose(gt_nonan_mask, output_nonan_mask, rtol=1e-02)


@pytest.mark.unit_tests
def test_create_mode_masks():
    """
    Test the _create_mode_masks function
    - Creates a map image for both sec and ref supports
    - Manually computes the standard, intersection and exclusion masks
    - Tests that the computed masks from _create_mode_masks
      are equal to ground truth
    """

    # Generate dsm with the following data and
    # "gironde_test_data" DSM's georef and resolution
    data = np.array([[1, 0, 1], [1, -9999, 1], [-1, 0, 1]], dtype=np.float32)
    data_dataset = dem_tools.create_dem(data=data, nodata=-9999)
    # Compute slope and add it as a classification_layer
    data_dataset = dem_tools.compute_dem_slope(data_dataset)
    # Classification layer configuration
    layer_name = "Slope0"
    clayer = {
        "type": "slope",
        "ranges": [0, 5, 10, 25, 45],
        "save_results": False,
        "output_dir": "",
        "nodata": -9999,
        "metrics": _DEFAULT_TEST_METRICS,
    }

    # Initialize slope classification layer object
    classif_layer_ = ClassificationLayer(
        name=layer_name,
        classification_layer_kind=str(clayer["type"]),
        dem=data_dataset,
        cfg=clayer,
    )

    # Initialize ground truth sec map_image.
    # input ranges are [0, 5, 10, 25, 45]
    dem_map_img = np.array(
        [[0.0, 0.0, 5.0], [5.0, 10.0, 10.0], [25.0, 45.0, 45.0]]
    )
    # Initialize ground truth ref map_image.
    ref_map_img = np.array(
        [[0.0, 5.0, 5.0], [5.0, 10.0, 10.0], [25.0, 45.0, 25.0]]
    )
    # Reset and set ground truth map_image on the classification_layer object
    classif_layer_.map_image["ref"] = ref_map_img
    classif_layer_.map_image["sec"] = dem_map_img
    # Reset and create sets_masks on the input map_image
    classif_layer_.classes_masks["ref"] = None
    classif_layer_.classes_masks["sec"] = None

    classif_layer_._create_class_masks()
    # Create mode_masks on the input map_image. We set the data_dataset as
    # input altitude dataset.
    mode_masks, _ = classif_layer_._create_mode_masks(alti_map=data_dataset)

    # standard mode will only have False where the input alti_map has
    # nodata value
    gt_standard_mode_mask = np.array(
        [[True, True, True], [True, False, True], [True, True, True]]
    )
    # intersection_mode will only have True where both classified_dem_slope
    # and classified_ref_slope
    # have the same class and input alti_map is not nodata
    gt_intersection_mode_mask = np.array(
        [[True, False, True], [True, False, True], [True, True, False]]
    )
    # exclusion_mode is the inverse of intersection_mode
    gt_exclusion_mode_mask = np.array(
        [[False, True, False], [False, False, False], [False, False, True]]
    )

    # Test that the computed masks_modes are the same as ground truth
    np.testing.assert_allclose(gt_standard_mode_mask, mode_masks[0], rtol=1e-02)
    np.testing.assert_allclose(
        gt_intersection_mode_mask, mode_masks[1], rtol=1e-02
    )
    np.testing.assert_allclose(
        gt_exclusion_mode_mask, mode_masks[2], rtol=1e-02
    )
