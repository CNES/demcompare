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
methods in the scalar metric class.
"""

# Third party imports
import numpy as np
import pytest

from demcompare.metric import Metric


@pytest.mark.unit_tests
def test_mean():
    """
    Test the mean metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)
    # Manually compute metric
    gt_output = np.mean(data)
    # Create metric object
    metric_obj = Metric("mean")
    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    assert gt_output == output


@pytest.mark.unit_tests
def test_max():
    """
    Test the max metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)
    # Manually compute metric
    gt_output = np.max(data)
    # Create metric object
    metric_obj = Metric("max")
    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    assert gt_output == output


@pytest.mark.unit_tests
def test_min():
    """
    Test the min metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)
    # Manually compute metric
    gt_output = np.min(data)
    # Create metric object
    metric_obj = Metric("min")
    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    assert gt_output == output


@pytest.mark.unit_tests
def test_std():
    """
    Test the std metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)
    # Manually compute metric
    gt_output = np.std(data)
    # Create metric object
    metric_obj = Metric("std")
    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    assert gt_output == output


@pytest.mark.unit_tests
def test_rmse():
    """
    Test the rmse metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)
    # Manually compute metric
    gt_output = np.sqrt(np.mean(data * data))
    # Create metric object
    metric_obj = Metric("rmse")
    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    assert gt_output == output


@pytest.mark.unit_tests
def test_median():
    """
    Test the median metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)
    # Manually compute metric
    gt_output = np.median(data)
    # Create metric object
    metric_obj = Metric("median")
    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    assert gt_output == output


@pytest.mark.unit_tests
def test_nmad():
    """
    Test the nmad metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)
    # Manually compute metric
    gt_output = 1.4826 * np.nanmedian(np.abs(data - np.nanmedian(data)))
    # Create metric object
    metric_obj = Metric("nmad")
    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    assert gt_output == output


@pytest.mark.unit_tests
def test_sum():
    """
    Test the sum_err metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)
    # Manually compute metric
    gt_output = np.sum(data)
    # Create metric object
    metric_obj = Metric("sum")
    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    assert gt_output == output


@pytest.mark.unit_tests
def test_percentil_90():
    """
    Test the percentil_90 metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """

    # Initialize input data
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)
    # Create metric object
    metric_obj = Metric("percentil_90")

    # Manually compute metric
    gt_output = np.nanpercentile(np.abs(data - np.nanmean(data)), 90)
    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    np.testing.assert_allclose(gt_output, output, rtol=1e-02)
