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
            [1914.0, 1895.0, 1880.0, 1857.0, 1872.0, 1890.0],
            [1901.0, 1885.0, 1869.0, 1855.0, 1871.0, 1888.0],
            [1892.0, 1875.0, 1858.0, 1857.0, 1878.0, 1900.0],
            [1371.0, 1351.0, 1320.0, 1761.0, 1764.0, 1759.0],
            [1355.0, 1342.0, 1317.0, 1760.0, 1760.0, 1758.0],
            [1343.0, 1321.0, 1294.0, 1760.0, 1757.0, 1754.0],
        ],
        dtype=np.float32,
    )
    ref = np.array(
        [
            [1913.0, 1898.0, 1879.0, 1855.0, 1871.0, 1891.0],
            [1905.0, 1890.0, 1872.0, 1855.0, 1873.0, 1894.0],
            [1890.0, 1876.0, 1861.0, 1855.0, 1874.0, 1898.0],
            [1374.0, 1349.0, 1305.0, 1761.0, 1761.0, 1760.0],
            [1358.0, 1343.0, 1311.0, 1759.0, 1760.0, 1758.0],
            [1357.0, 1342.0, 1310.0, 1757.0, 1756.0, 1754.0],
        ],
        dtype=np.float32,
    )

    sec_dataset = dem_tools.create_dem(data=sec, nodata=-37)
    ref_dataset = dem_tools.create_dem(data=ref, nodata=99)

    # Define ground truth value
    diff_gt = np.array(
        [
            [
                -2.13889,
                1.8611128,
                -2.1388907,
                -3.1388893,
                -2.1388905,
                -0.1388889,
            ],
            [2.861113, 3.861115, 1.8611127, -1.138889, 0.86111194, 4.8611135],
            [
                -3.1389298,
                -0.13889067,
                1.8611313,
                -3.1389077,
                -5.138903,
                -3.1388984,
            ],
            [
                1.8611357,
                -3.1389303,
                -16.139097,
                -1.1389,
                -4.1389008,
                -0.1388893,
            ],
            [
                1.8611127,
                -0.13888901,
                -7.138963,
                -2.1389124,
                -1.138889,
                -1.138889,
            ],
            [12.861118, 19.861135, 14.86127, -4.1389356, -2.138889, -1.138889],
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
            [19.547047, 19.55331, 19.560884, 19.568203, 19.552214, 19.561977],
            [19.63805, 20.045979, 20.727287, 20.874863, 20.879704, 21.060541],
            [21.295753, 21.389414, 21.219566, 21.058632, 21.13076, 20.918678],
            [20.903389, 20.744022, 20.70011, 20.294266, 19.87637, 19.726604],
            [20.337505, 20.917957, 21.017256, 21.004324, 21.197472, 21.304148],
            [21.458612, 21.55021, 22.57449, 22.878378, 23.041908, 23.01843],
        ],
        dtype=np.float32,
    )
    ref = np.array(
        [
            [19.31176, 19.22148, 19.211123, 19.293081, 19.26587, 19.203588],
            [19.211302, 19.3398, 19.644583, 19.965935, 20.91119, 20.868784],
            [20.847872, 20.863094, 20.843157, 20.69857, 20.411663, 20.092188],
            [19.784042, 19.51518, 19.33782, 19.233442, 19.191921, 19.214354],
            [19.346697, 19.64496, 20.012804, 20.353899, 20.642817, 20.8117],
            [20.90534, 20.999607, 20.481012, 20.4837, 20.482937, 20.48312],
        ],
        dtype=np.float32,
    )

    sec_dataset = dem_tools.create_dem(data=sec, nodata=-33)
    ref_dataset = dem_tools.create_dem(data=ref)

    # Define ground truth value

    diff_gt = np.array(
        [
            [
                0.59798896,
                0.5014447,
                0.48351377,
                0.55815405,
                0.54693127,
                0.4748869,
            ],
            [
                0.4065275,
                0.127097,
                -0.24942902,
                -0.07565233,
                0.8647624,
                0.64151865,
            ],
            [
                0.385394,
                0.30695614,
                0.4568661,
                0.47321403,
                0.11417851,
                0.00678522,
            ],
            [
                -0.28607106,
                -0.39556623,
                -0.5290139,
                -0.22754785,
                0.14882739,
                0.32102475,
            ],
            [
                -0.15753289,
                -0.4397214,
                -0.1711762,
                0.18285076,
                0.27862075,
                0.34082896,
            ],
            [
                0.28000343,
                0.28267184,
                -1.2602022,
                -1.5614032,
                -1.7256964,
                -1.7020358,
            ],
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
