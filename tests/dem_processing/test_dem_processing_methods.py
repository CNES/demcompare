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
This module contains functions to test the
methods in the DEM processing class.
"""

# Third party imports
import numpy as np
import pytest

from demcompare import dem_tools
from demcompare.dem_processing import DemProcessing


@pytest.mark.unit_tests
def test_alti_diff():
    """
    Test alti-diff DEM processing class function process_dem.
    Input data:
    - Two manually created dems with custom nodata (-37, 99, 33)
      values
    Validation data:
    - Manually computed dem diff: diff_gt
    Validation process:
    - Create both dems
    - Compute the difference dem using the process_dem function
    - Check that the difference dem is the same as ground truth
    - Checked function : dem_tools's process_dem
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

    dem_processing_obj = DemProcessing("alti-diff")
    diff_dataset = dem_processing_obj.process_dem(ref_dataset, sec_dataset)
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
            [dem_tools.DEFAULT_NODATA, 0, 1],
            [1, 1, 0],
            [1, 1, dem_tools.DEFAULT_NODATA],
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
    diff_dataset = dem_processing_obj.process_dem(ref_dataset, sec_dataset)
    # Test that the output difference is the same as ground_truth
    np.testing.assert_array_equal(diff_gt, diff_dataset["image"].data)


@pytest.mark.unit_tests
def test_angular_diff():
    """
    Test angular-diff DEM processing class function process_dem.
    Input data:
    - Two manually created dems with custom nodata (-37, 99, 33)
      values
    Validation data:
    - Manually computed dem diff: diff_gt
    Validation process:
    - Create both dems
    - Compute the difference dem using the process_dem function
    - Check that the difference dem is the same as ground truth
    - Checked function : dem_tools's process_dem
    """
    # Create input datasets
    sec = np.array(
        [
            [1982.0, 1967.0, 1950.0],
            [2005.0, 1988.0, 1969.0],
            [2012.0, 1990.0, 1967.0],
        ],
        dtype=np.float32,
    )
    ref = np.array(
        [
            [1913.0, 1898.0, 1879.0],
            [1905.0, 1890.0, 1872.0],
            [1890.0, 1876.0, 1861.0],
        ],
        dtype=np.float32,
    )

    sec_dataset = dem_tools.create_dem(data=sec, nodata=-37)
    ref_dataset = dem_tools.create_dem(data=ref, nodata=99)

    # Define ground truth value
    diff_gt = np.array(
        [
            [1.480917, 1.357961, 1.192593],
            [1.375154, 1.154992, 0.883184],
            [1.126594, 0.855523, 0.545449],
        ],
        dtype=np.float32,
    )

    dem_processing_obj = DemProcessing("angular-diff")
    diff_dataset = dem_processing_obj.process_dem(ref_dataset, sec_dataset)
    # Test that the output difference is the same as ground_truth
    np.testing.assert_array_almost_equal(diff_gt, diff_dataset["image"].data)

    # Create input datasets
    sec = np.array(
        [[1, 1, 1], [1, 1, 1], [-1, -33, 1], [1, 2, -33], [1, 1, -33]],
        dtype=np.float32,
    )
    ref = np.array(
        [
            [3, 3, 3],
            [1, 2, 1],
            [dem_tools.DEFAULT_NODATA, 0, 1],
            [1, 1, 0],
            [1, 1, dem_tools.DEFAULT_NODATA],
        ],
        dtype=np.float32,
    )

    sec_dataset = dem_tools.create_dem(data=sec, nodata=-33)
    ref_dataset = dem_tools.create_dem(data=ref)

    # Define ground truth value

    diff_gt = np.array(
        [
            [1.107149, 0.785398, 1.107149],
            [np.nan, np.nan, 0.955317],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [0.0, np.nan, np.nan],
        ],
        dtype=np.float32,
    )

    diff_dataset = dem_processing_obj.process_dem(ref_dataset, sec_dataset)
    # Test that the output difference is the same as ground_truth
    np.testing.assert_array_almost_equal(diff_gt, diff_dataset["image"].data)


@pytest.mark.unit_tests
def test_alti_diff_norm():
    """
    Test alti-diff-slope-norm DEM processing class function process_dem.
    Input data:
    - Two manually created dems with custom nodata (-37, 99, 33)
      values
    Validation data:
    - Manually computed dem diff: diff_gt
    Validation process:
    - Create both dems
    - Compute the difference dem using the process_dem function
    - Check that the difference dem is the same as ground truth
    - Checked function : dem_tools's process_dem
    """
    # Create input datasets
    sec = np.array(
        [
            [1982.0, 1967.0, 1950.0],
            [2005.0, 1988.0, 1969.0],
            [2012.0, 1990.0, 1967.0],
        ],
        dtype=np.float32,
    )
    ref = np.array(
        [
            [1913.0, 1898.0, 1879.0],
            [1905.0, 1890.0, 1872.0],
            [1890.0, 1876.0, 1861.0],
        ],
        dtype=np.float32,
    )

    sec_dataset = dem_tools.create_dem(data=sec, nodata=-37)
    ref_dataset = dem_tools.create_dem(data=ref, nodata=99)

    # Define ground truth value
    diff_gt = np.array(
        [
            [-0.032235, -0.033529, -0.031944],
            [0.009373, 0.006633, 0.005105],
            [0.042951, 0.031357, 0.018408],
        ],
        dtype=np.float32,
    )

    dem_processing_obj = DemProcessing("alti-diff-slope-norm")
    diff_dataset = dem_processing_obj.process_dem(ref_dataset, sec_dataset)
    # Test that the output difference is the same as ground_truth
    np.testing.assert_array_almost_equal(diff_gt, diff_dataset["image"].data)

    # Create input datasets
    sec = np.array(
        [
            [1761.0, 1764.0, 1759.0],
            [1760.0, 1760.0, 1758.0],
            [1760.0, 1757.0, 1754.0],
        ],
        dtype=np.float32,
    )
    ref = np.array(
        [
            [1761.0, 1761.0, 1760.0],
            [1759.0, 1760.0, 1758.0],
            [1757.0, 1756.0, 1754.0],
        ],
        dtype=np.float32,
    )

    sec_dataset = dem_tools.create_dem(data=sec, nodata=-33)
    ref_dataset = dem_tools.create_dem(data=ref)

    # Define ground truth value

    diff_gt = np.array(
        [
            [-0.09173357, 0.19564123, -0.12446043],
            [0.44559613, -0.07847814, -0.09047703],
            [0.2780473, 0.01896586, -0.05560689],
        ],
        dtype=np.float32,
    )
    diff_dataset = dem_processing_obj.process_dem(ref_dataset, sec_dataset)
    # Test that the output difference is the same as ground_truth
    np.testing.assert_array_almost_equal(diff_gt, diff_dataset["image"].data)


@pytest.mark.unit_tests
def test_ref_curvature():
    """
    Test ref-curvature DEM processing class function process_dem.
    Input data:
    - One manually created dem with custom nodata (-37, 99, 33)
      values
    Validation data:
    - Manually computed dem curvature: gt
    Validation process:
    - Create the dem
    - Compute the curvature dem using the process_dem function
    - Check that the curvature dem is the same as ground truth
    - Checked function : dem_tools's process_dem
    """
    # Create input dataset
    ref = np.array(
        [[3, 99, 3], [99, 2, 1], [99, 0, 1], [1, 1, 0], [1, 1, 99]],
        dtype=np.float32,
    )

    ref_dataset = dem_tools.create_dem(data=ref, nodata=99)

    # Define ground truth value
    gt = np.array(
        [
            [0.9398802, np.nan, 1.7072412],
            [np.nan, 0.7338672, -1.4214045],
            [np.nan, -2.4205306, 0.42490682],
            [0.36860558, 0.6654598, -1.5057596],
            [-0.0843266, 0.36212376, np.nan],
        ],
        dtype=np.float32,
    )

    dem_processing_obj = DemProcessing("ref-curvature")
    dataset = dem_processing_obj.process_dem(ref_dataset)

    # Test that the output difference is the same as ground_truth
    np.testing.assert_array_almost_equal(gt, dataset["image"].data)

    # Create input dataset
    ref = np.array(
        [
            [3, 3, 3],
            [1, 2, 1],
            [dem_tools.DEFAULT_NODATA, 0, 1],
            [1, 1, 0],
            [1, 1, dem_tools.DEFAULT_NODATA],
        ],
        dtype=np.float32,
    )
    ref_dataset = dem_tools.create_dem(data=ref)

    # Define ground truth value

    gt = np.array(
        [
            [1.7635386, 1.500691, 1.7770832],
            [-1.3848146, 1.3327632, -1.3950232],
            [np.nan, -2.5563521, 0.48228034],
            [-0.05303898, 0.64098966, -1.4813623],
            [-0.02683339, 0.37867376, np.nan],
        ],
        dtype=np.float32,
    )

    dataset = dem_processing_obj.process_dem(ref_dataset)

    # Test that the output curvature is the same as ground_truth
    np.testing.assert_array_almost_equal(gt, dataset["image"].data)
