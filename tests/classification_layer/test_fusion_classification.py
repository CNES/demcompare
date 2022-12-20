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
