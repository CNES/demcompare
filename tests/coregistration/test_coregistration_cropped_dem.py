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
# Disable the protected-access to test the functions

"""
This module contains functions to test
the coregistration.crop_dem_with_offset function
"""
# pylint:disable=protected-access
# pylint:disable=duplicate-code

# Third party imports
import numpy as np
import pytest


@pytest.mark.unit_tests
def test_crop_dem_with_offset_pos_x_pos_y(initialize_dem_and_coreg):
    """
    Test the crop_dem_with_offset function with
    positive x and positive y offsets.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually cropped input dem with the corresponding offsets: gt_cropped_dem.
    Validation process:
    - Crops the input dem with the crop_dem_with_offset function.
    - Checks that the obtained dem is the same as ground truth.
        - Checked function: coregistration.crop_dem_with_offset
    """
    coregistration_, input_dem = initialize_dem_and_coreg

    # Test with positive x_offset and positive y_offset
    x_offset = 2.3
    y_offset = 4.7
    output_cropped_dem = coregistration_.crop_dem_with_offset(
        input_dem, x_offset, y_offset
    )
    gt_cropped_dem = input_dem[
        int(np.floor(y_offset)) : input_dem.shape[0],
        0 : input_dem.shape[1] - int(np.ceil(x_offset)),
    ]
    np.testing.assert_allclose(gt_cropped_dem, output_cropped_dem, rtol=1e-02)


@pytest.mark.unit_tests
def test_crop_dem_with_offset_pos_x_neg_y(initialize_dem_and_coreg):
    """
    Test the crop_dem_with_offset function with
    positive x and negative y offsets.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually cropped input dem with the corresponding offsets: gt_cropped_dem.
    Validation process:
    - Crops the input dem with the crop_dem_with_offset function.
    - Checks that the obtained dem is the same as ground truth.
        - Checked function: coregistration.crop_dem_with_offset
    """
    coregistration_, input_dem = initialize_dem_and_coreg

    # Test with positive x_offset and negative y_offset
    x_offset = 2.3
    y_offset = -4.7
    output_cropped_dem = coregistration_.crop_dem_with_offset(
        input_dem, x_offset, y_offset
    )
    gt_cropped_dem = input_dem[
        0 : input_dem.shape[0] - int(np.ceil(-y_offset)),
        0 : input_dem.shape[1] - int(np.ceil(x_offset)),
    ]
    np.testing.assert_allclose(gt_cropped_dem, output_cropped_dem, rtol=1e-02)


@pytest.mark.unit_tests
def test_crop_dem_with_offset_neg_x_pos_y(initialize_dem_and_coreg):
    """
    Test the crop_dem_with_offset function with
    negative x and positive y offsets.
     Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually cropped input dem with the corresponding offsets: gt_cropped_dem.
    Validation process:
    - Crops the input dem with the crop_dem_with_offset function.
    - Checks that the obtained dem is the same as ground truth.
        - Checked function: coregistration.crop_dem_with_offset
    """
    coregistration_, input_dem = initialize_dem_and_coreg

    # Test with negative x_offset and positive y_offset
    x_offset = -2.3
    y_offset = 4.7
    output_cropped_dem = coregistration_.crop_dem_with_offset(
        input_dem, x_offset, y_offset
    )
    gt_cropped_dem = input_dem[
        int(np.floor(y_offset)) : input_dem.shape[0],
        int(np.floor(-x_offset)) : input_dem.shape[1],
    ]
    np.testing.assert_allclose(gt_cropped_dem, output_cropped_dem, rtol=1e-02)


@pytest.mark.unit_tests
def test_crop_dem_with_offset_neg_x_neg_y(initialize_dem_and_coreg):
    """
    Test the crop_dem_with_offset function with
    negative x and negative y offsets.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually cropped input dem with the corresponding offsets: gt_cropped_dem.
    Validation process:
    - Crops the input dem with the crop_dem_with_offset function.
    - Checks that the obtained dem is the same as ground truth.
        - Checked function: coregistration.crop_dem_with_offset
    """
    coregistration_, input_dem = initialize_dem_and_coreg

    # Test with negative x_offset and negative y_offset
    x_offset = -2.3
    y_offset = -4.7
    output_cropped_dem = coregistration_.crop_dem_with_offset(
        input_dem, x_offset, y_offset
    )
    gt_cropped_dem = input_dem[
        0 : input_dem.shape[0] - int(np.ceil(-y_offset)),
        int(np.floor(-x_offset)) : input_dem.shape[1],
    ]
    np.testing.assert_allclose(gt_cropped_dem, output_cropped_dem, rtol=1e-02)
