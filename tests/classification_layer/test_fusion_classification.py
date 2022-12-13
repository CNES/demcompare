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
methods in the fusion classification layer class.
"""

from collections import OrderedDict

# Third party imports
import numpy as np
import pytest
import xarray as xr

# Demcompare imports
from demcompare import dem_tools
from demcompare.classification_layer.classification_layer import (
    ClassificationLayer,
)
from demcompare.classification_layer.fusion_classification import (
    FusionClassificationLayer,
)


@pytest.fixture(name="initialize_fusion_layer")
def fixture_initialize_fusion_layer():
    """
    Fixture to initialize the fusion layer
    - Manually creates two input dems (dataset_ref and dataset_sec)
      with an image array
    - The dataset_sec dem contains
      one classification_layer_mask, called "seg_classif"
    - Computes the slope of both dems using the "compute_dem_slope" function
    - Computes the altitude difference between both dems
    - Creates a segmentation classification layer created using
      altitude difference dataset
      The classification layer is called "seg_classif".
    - Creates a slope classification layer using the altitude difference
      dataset
    - Forces the map_image["sec"] of the slope classification to be
      a manually created array
    - Creates a Fusion classification layer using the segmentation and
      the slope classification layers with sec support
    - Returns the created fusion classification layer object
    """
    data = np.array(
        [[0, 1, 1], [0, -9999, 1], [-9999, 1, 1], [-9999, 1, 1]],
        dtype=np.float32,
    )

    # Classification layer configuration
    seg_name = "seg_classif"
    seg_clayer = {
        "type": "segmentation",
        "classes": {"sea": [0], "deep_land": [1]},
        "output_dir": "",
        "nodata": -9999,
        "metrics": ["mean"],
    }

    # Create the ref dataset
    dataset_ref = dem_tools.create_dem(data=data)

    # Create the sec dataset

    # Initialize the data of the classification layers
    classif_data = np.full((data.shape[0], data.shape[1], 1), np.nan)
    classif_data[:, :, 0] = np.array(
        [[0, 1, 1], [1, 1, 0], [-9999, 1, 1], [-9999, 1, 0]]
    )
    classif_name = ["seg_classif"]

    # Initialize the coordinates of the classification layers
    coords_classification_layers = [
        dataset_ref.coords["row"],
        dataset_ref.coords["col"],
        classif_name,
    ]
    # Create the datarray with input classification layers
    # for the sec dataset
    seg_classif_layer_mask = xr.DataArray(
        data=classif_data,
        coords=coords_classification_layers,
        dims=["row", "col", "indicator"],
    )

    # Create the sec dataset
    dataset_sec = dem_tools.create_dem(
        data=data, classification_layer_masks=seg_classif_layer_mask
    )

    # Compute slope and add it as a classification_layer
    dataset_ref = dem_tools.compute_dem_slope(dataset_ref)
    dataset_sec = dem_tools.compute_dem_slope(dataset_sec)
    # Compute altitude diff
    final_altitude_diff = dem_tools.compute_alti_diff_for_stats(
        dataset_ref, dataset_sec
    )

    # Initialize classification layer object
    seg_classif_layer_ = ClassificationLayer(
        name=seg_name,
        classification_layer_kind=str(seg_clayer["type"]),
        dem=final_altitude_diff,
        cfg=seg_clayer,
    )

    # Classification layer configuration
    slope_name = "Slope0"
    slope_clayer = {
        "type": "slope",
        "ranges": [0, 5, 10],
        "output_dir": "",
        "nodata": -9999,
        "metrics": ["mean"],
    }

    # Initialize slope classification layer object
    slope0_classif_layer_ = ClassificationLayer(
        name=slope_name,
        classification_layer_kind=str(slope_clayer["type"]),
        dem=final_altitude_diff,
        cfg=slope_clayer,
    )

    # Initialize ground truth sec map_image.
    # input ranges are [0, 5, 10, 25, 45]
    slope_dem_map_img = np.array(
        [
            [0.0, 0.0, 5.0],
            [5.0, 10.0, 10.0],
            [0.0, -9999, -9999],
            [0.0, 10.0, 5.0],
        ]
    )
    # Set ground truth map_image on the classification_layer object
    # It will be placed on idx 1
    slope0_classif_layer_.map_image["sec"] = slope_dem_map_img
    # Create sets_masks on the input map_image
    slope0_classif_layer_._create_class_masks()
    # Create a fusion layer with the two input classes
    # Since we want to fusion sec images, map_idx is 1
    fusion_layer_ = FusionClassificationLayer(
        [seg_classif_layer_, slope0_classif_layer_],
        support="sec",
        name="Fusion0",
        metrics=["mean"],
    )
    return fusion_layer_


@pytest.mark.unit_tests
def test_create_merged_classes(initialize_fusion_layer):
    """
    Test the _create_merged_classes function
    Input data:
    - Fusion classification layer from the "initialize_fusion_layer"
      fixture
    Validation data:
    - Manually computed dictionary for the merged classes: gt_merged_classes
    Validation process:
    - Compute the fusion classification layer
    - Check that the computed classes from _create_merged_classes
      are equal to ground truth
    - Checked function : FusionClassificationLayer's _create_merged_classes
    - Checked attribute : ClassificationLayer's classes
    """
    fusion_layer_ = initialize_fusion_layer
    # Test the _create_merged_classes function ------------------
    # All possible classes combination
    # Slope0 layer ranges are [0, 5, 10], so its classes are
    #     OrderedDict([('[0%;5%[', 0), ('[5%;10%[', 5), ('[10%;inf[', 10)])
    # seg_classif layer classes are [  'sea',  'deep_land'  ]
    #     OrderedDict([('sea', [0]), ('deep_land', [1])])

    gt_merged_classes = OrderedDict(
        [
            ("seg_classif_sea_&_Slope0_[0%;5%[", 1),
            ("seg_classif_sea_&_Slope0_[5%;10%[", 2),
            ("seg_classif_sea_&_Slope0_[10%;inf[", 3),
            ("seg_classif_deep_land_&_Slope0_[0%;5%[", 4),
            ("seg_classif_deep_land_&_Slope0_[5%;10%[", 5),
            ("seg_classif_deep_land_&_Slope0_[10%;inf[", 6),
        ]
    )
    # Test that the fusion layer's classes as the same as gt
    assert fusion_layer_.classes == gt_merged_classes


@pytest.mark.unit_tests
def test_merge_classes_and_create_sets_masks(initialize_fusion_layer):
    """
    Test the _merge_classes_and_create_sets_masks function
    Input data:
    - Fusion classification layer from the "initialize_fusion_layer"
      fixture
    Validation data:
    - Manually computed classes masks: gt_sets_masks
    Validation process:
    - Compute the fusion classification layer
    - Check that the computed classes masks from
      _merge_classes_and_create_sets_masks
      are equal to ground truth
    - Checked function : FusionClassificationLayer's
      _merge_classes_and_create_sets_masks
    - Checked attribute : ClassificationLayer's classes_masks
    """
    # Test the _merge_classes_and_create_sets_masks function -------------
    # Slope0's map_image["Slope0"] is :
    #   array([[ 0,  0,  5],
    #          [ 5,  10,  10],
    #          [ 0, -9999, -9999],
    #          [ 0,  10,  5]])
    # seg_classif's map_image["Status"] is :
    #   array([[ 0,  1,  1],
    #          [ 1,  1,  0],
    #          [-9999,  1,  1],
    #          [-9999,  1,  0]])
    fusion_layer_ = initialize_fusion_layer

    # Create ground truth sets masks dict for the input classified slope
    # fusion layer's class 1
    mask_range_seg_0_slope_0 = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    # fusion layer's class 2
    mask_range_seg_0_slope_5 = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    # fusion layer's class 3
    mask_range_seg_0_slope_10 = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    # fusion layer's class 4
    mask_range_seg_1_slope_0 = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    # fusion layer's class 5
    mask_range_seg_1_slope_5 = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    # fusion layer's class 6
    mask_range_seg_1_slope_10 = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    gt_sets_masks = [
        mask_range_seg_0_slope_0,
        mask_range_seg_0_slope_5,
        mask_range_seg_0_slope_10,
        mask_range_seg_1_slope_0,
        mask_range_seg_1_slope_5,
        mask_range_seg_1_slope_10,
    ]

    gt_classes_masks = {
        "fusion_layer": gt_sets_masks,
    }
    # Test that the computed sets_masks_dict is the same as gt
    np.testing.assert_allclose(
        gt_classes_masks["fusion_layer"],
        fusion_layer_.classes_masks["sec"],
        rtol=1e-02,
    )


@pytest.mark.unit_tests
def test_create_labelled_map(initialize_fusion_layer):
    """
    Test the _create_labelled_map function
    Input data:
    - Fusion classification layer from the "initialize_fusion_layer"
      fixture
    Validation data:
    - Manually computed map_image: gt_map_image
    Validation process:
    - Compute the fusion classification layer
    - Check that the computed map_image from
      _create_labelled_map
      is equal to ground truth
    - Checked function : ClassificationLayer's _create_labelled_map
    - Checked attribute : ClassificationLayer's map_image
    """
    fusion_layer_ = initialize_fusion_layer
    # Slope0's map_image["Slope0"] is :
    #   array([[ 0,  0,  5],
    #          [ 5,  10,  10],
    #          [ 0, -9999, -9999],
    #          [ 0,  10,  5]])
    # seg_classif's map_image["Status"] is :
    #   array([[ 0,  1,  1],
    #          [ 1,  1,  0],
    #          [-9999,  1,  1],
    #          [-9999,  1,  0]])
    # The merged classes are
    # ("seg_classif_sea_&_Slope0_[0%;5%[", 1),
    # ("seg_classif_sea_&_Slope0_[5%;10%[", 2),
    # ("seg_classif_sea_&_Slope0_[10%;inf[", 3),
    # ("seg_classif_deep_land_&_Slope0_[0%;5%[", 4),
    # ("seg_classif_deep_land_&_Slope0_[5%;10%[", 5),
    # ("seg_classif_deep_land_&_Slope0_[10%;inf[", 6),

    # test the _create_labelled_map function -------------
    # fusion layer's classified map image
    gt_map_image = np.array(
        [[1, 4, 5], [5, 6, 3], [-9999, -9999, -9999], [-9999, 6, 2]]
    )

    np.testing.assert_allclose(
        gt_map_image, fusion_layer_.map_image["sec"], rtol=1e-02
    )
