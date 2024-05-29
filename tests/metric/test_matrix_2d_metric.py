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
methods in the matrix 2D metric class.
"""
# pylint:disable=protected-access
# Third party imports
import numpy as np
import pytest

from demcompare.metric import Metric
from tests.helpers import RESULT_TOL


@pytest.mark.unit_tests
def test_svf():
    """
    Test the svf metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output, gt_bins and gt= with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    # Test without percentil 98 filtering ---------------------------

    # Initialize input data
    data = np.array(
        [
            [1982.0, 1967.0, 1950.0],
            [2005.0, 1988.0, 1969.0],
            [2012.0, 1990.0, 1967.0],
        ],
        dtype=np.float32,
    )

    # Create metric object
    metric_obj = Metric("svf")

    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    gt = np.array(
        [
            [223.102478, 72.567892, 0.0],
            [255.0, 255.0, 185.740152],
            [255.0, 255.0, 46.564479],
        ]
    )

    np.testing.assert_allclose(output["image"].data, gt, rtol=RESULT_TOL)


@pytest.mark.unit_tests
def test_hillshade():
    """
    Test the hillshade metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output, gt_bins and gt= with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    # Test without percentil 98 filtering ---------------------------

    # Initialize input data
    data = np.array(
        [
            [1982.0, 1967.0, 1950.0],
            [2005.0, 1988.0, 1969.0],
            [2012.0, 1990.0, 1967.0],
        ],
        dtype=np.float32,
    )

    # Create metric object
    metric_obj = Metric("hillshade")

    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    gt = np.array(
        [
            [149.34193, 142.97736, 136.03064],
            [125.85443, 112.33784, 99.7046],
            [90.020424, 73.68868, 62.433144],
        ]
    )

    np.testing.assert_allclose(output["image"].data, gt, rtol=RESULT_TOL)
