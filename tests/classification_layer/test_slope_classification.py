#!/usr/bin/env python
# coding: utf8
# Disable the protected-access to test the functions
# pylint:disable=protected-access
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
This module contains functions to test the
methods in the slope_classification class.
"""

# Third party imports
import numpy as np
import pytest

# Demcompare imports
from demcompare import dem_tools
from demcompare.classification_layer.classification_layer import (
    ClassificationLayer,
)
from demcompare.dem_tools import create_dem


@pytest.fixture(name="initialize_slope_layer")
def fixture_initialize_slope_layer():
    """
    Fixture to initialize a slope layer
    """
    # Generate dsm with the following data and
    # "gironde_test_data" DSM's georef and resolution
    data = np.array([[1, 0, 1], [1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    dem_dataset = dem_tools.create_dem(data=data)
    # Compute dem's slope
    dem_dataset = dem_tools.compute_dem_slope(dem_dataset)
    # Classification layer configuration
    layer_name = "Slope0"
    clayer = {
        "type": "slope",
        "ranges": [0, 5, 10, 25, 45],
        "output_dir": "",
        "metrics": ["mean"],
    }

    # Initialize slope classification layer object
    slope_classif_layer_ = ClassificationLayer(
        name=layer_name,
        classification_layer_kind=str(clayer["type"]),
        dem=dem_dataset,
        cfg=clayer,
    )
    return slope_classif_layer_, dem_dataset


@pytest.mark.unit_tests
def test_classify_slope_by_ranges(initialize_slope_layer):
    """
    Test the classify_slope_by_ranges function
    - Creates a slope dem
    - Manually classifies the slope dem with the input ranges
    - Tests that the classified slope by the function
      classify_slope_by_ranges is the same as ground truth
    """
    slope_classif_layer_, dem_dataset = initialize_slope_layer
    # Initialize slope array
    slope_image = np.array([[1, 0.0, 6], [8, 12, 20], [26, 50, 60]])
    # Initialize slope dem
    slope_dataset = create_dem(
        slope_image,
        transform=dem_dataset.georef_transform.data,
        img_crs=dem_dataset.crs,
    )

    # Obtain output classified slope by ranges of the input slope_dataset
    slope_classif_layer_._classify_slope_by_ranges(slope_dataset)
    output_classified_slope = slope_classif_layer_.map_image["ref"]

    # Ground truth classified slope by ranges
    # The input range is [0, 5, 10, 25, 45]
    gt_classified_slope = np.array(
        [[0.0, 0.0, 5.0], [5.0, 10.0, 10.0], [25.0, 45.0, 45.0]]
    )

    # Test that the output classified slope is the same as gt
    np.testing.assert_allclose(
        gt_classified_slope, output_classified_slope, rtol=1e-02
    )


@pytest.mark.unit_tests
def test_create_class_masks(initialize_slope_layer):
    """
    Test the _create_class_masks function
    - Creates a slope dem
    - Manually classifies the slope dem with the input ranges
    - Tests that the computed sets_masks_dict is equal to ground truth
    """
    slope_classif_layer_, _ = initialize_slope_layer

    # Initialize slope array
    # slope_image = np.array([[1, 0.0, 6], [8, 12, 20], [26, 50, 60]])
    # Initialize ground truth classified slope by ranges [0, 5, 10, 25, 45]
    classified_slope = np.array(
        [[0.0, 0.0, 5.0], [5.0, 10.0, 10.0], [25.0, 45.0, 45.0]]
    )
    # Add ground truth classified slope by ranges
    # The input range is [0, 5, 10, 25, 45]
    slope_classif_layer_.map_image["ref"] = classified_slope
    # Create sets_masks on the input classified slope
    slope_classif_layer_._create_class_masks()
    # Create ground truth sets masks dict for the input classified slope
    mask_range_0 = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mask_range_5 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mask_range_10 = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    )
    mask_range_25 = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    mask_range_45 = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
    )
    gt_classes_masks_masks = [
        mask_range_0,
        mask_range_5,
        mask_range_10,
        mask_range_25,
        mask_range_45,
    ]

    # Test that the computed sets_masks_dict is the same as ground truth
    np.testing.assert_allclose(
        gt_classes_masks_masks,
        slope_classif_layer_.classes_masks["ref"],
        rtol=1e-02,
    )
