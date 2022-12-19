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
This module contains functions to test the translate_dem function.
"""
# pylint:disable = duplicate-code
# Standard imports

# Third party imports
import numpy as np
import pytest

# Demcompare imports
from demcompare import dem_tools

# Force protected access to test protected functions
# pylint:disable=protected-access


@pytest.mark.unit_tests
def test_translate_dem_pos_x_neg_y(load_gironde_dem):
    """
    Test the translate_dem function with positive x
    and negative y offsets.
    Input data:
    - Sec dem from the "load_gironde_dem" fixture.
    Validation data:
    - The dem's georef_transform's values at position
      0 and 3 when the offset has been applied: gt_offset_coord_x,
      gt_offset_coord_y
    Validation process:
    - Apply a pixellic offset (positive x, negative y)
      to the dem using the translate_dem function
    - Check that the translated dem's georef_transform
      positions 0 and 3 are the same as the ground truth offsets
    - Checked function : dem_tools's translate_dem
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
    Input data:
    - Sec dem from the "load_gironde_dem" fixture.
    Validation data:
    - The dem's georef_transform's values at position
      0 and 3 when the offset has been applied: gt_offset_coord_x,
      gt_offset_coord_y
    Validation process:
    - Apply a pixellic offset (negative x, negative y)
      to the dem using the translate_dem function
    - Check that the translated dem's georef_transform
      positions 0 and 3 are the same as the ground truth offsets
    - Checked function : dem_tools's translate_dem
    """
    # When translate_dem, the values at position 0 and 3 of the
    # transform are modified
    # by the offset coordinates x and y
    # Original dem transform 0 and 3 position values before translate_dem :
    # trans[0] = 600255
    # trans[3] = 5099745

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
    Input data:
    - Sec dem from the "load_gironde_dem" fixture.
    Validation data:
    - The dem's georef_transform's values at position
      0 and 3 when the offset has been applied: gt_offset_coord_x,
      gt_offset_coord_y
    Validation process:
    - Apply a pixellic offset (negative x, positive y)
      to the dem using the translate_dem function
    - Check that the translated dem's georef_transform
      positions 0 and 3 are the same as the ground truth offsets
    - Checked function : dem_tools's translate_dem
    """
    # When translate_dem, the values at position 0 and 3 of the
    # transform are modified
    # by the offset coordinates x and y
    # Original dem transform 0 and 3 position values before translate_dem :
    # trans[0] = 600255
    # trans[3] = 5099745

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
    Input data:
    - Sec dem from the "load_gironde_dem" fixture.
    Validation data:
    - The dem's georef_transform's values at position
      0 and 3 when the offset has been applied: gt_offset_coord_x,
      gt_offset_coord_y
    Validation process:
    - Apply a pixellic offset (positive x, positive y)
      to the dem using the translate_dem function
    - Check that the translated dem's georef_transform
      positions 0 and 3 are the same as the ground truth offsets
    - Checked function : dem_tools's translate_dem
    """
    # When translate_dem, the values at position 0 and 3 of the
    # transform are modified
    # by the offset coordinates x and y
    # Original dem transform 0 and 3 position values before translate_dem :
    # trans[0] = 600255
    # trans[3] = 5099745

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
