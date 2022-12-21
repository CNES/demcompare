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
methods in the vector metric class.
"""
# pylint:disable=protected-access
# Third party imports
import numpy as np
import pytest

from demcompare.metric import Metric


@pytest.mark.unit_tests
def test_ratio_above_threshold():
    """
    Test the ratio_above_threshold metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output and gt_ratio with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """

    # Test with default elevation threshold ----------------------------
    # Initialize input data
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)
    # Create metric object
    metric_obj = Metric("ratio_above_threshold")
    # Define elevation thresholds as default
    elevation_thrlds = metric_obj._ELEVATION_THRESHOLDS

    # Manually compute metric
    gt_ratio = []
    for threshold in elevation_thrlds:
        gt_ratio.append(
            ((np.count_nonzero(data > threshold)) / float(data.size))
        )
    gt_output = (gt_ratio, elevation_thrlds)
    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    np.testing.assert_allclose(gt_output, output, rtol=1e-02)

    # Test with custom elevation threshold ----------------------------

    elevation_thrlds = [-3, 2, 90]
    # Create metric object with custom parameter
    metric_obj = Metric(
        "ratio_above_threshold",
        params={"elevation_threshold": elevation_thrlds},
    )

    # Manually compute metric
    gt_ratio = []
    for threshold in elevation_thrlds:
        gt_ratio.append(
            ((np.count_nonzero(data > threshold)) / float(data.size))
        )
    gt_output = (gt_ratio, elevation_thrlds)

    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    np.testing.assert_allclose(gt_output, output, rtol=1e-02)  # type:ignore


@pytest.mark.unit_tests
def test_cdf():
    """
    Test the cdf metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output, gt_bins_count and gt_cdf
    with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """

    # Initialize input data
    data = np.array(
        [
            [-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0],
            [-1.0, 3.0, 4.0, 5, 3.0, -6.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Create metric object
    metric_obj = Metric("cdf")
    # Manually compute metric
    # Get max diff from data
    max_diff = np.nanmax(np.abs(data))
    # Get bins number for histogram
    bin_step = metric_obj._BIN_STEP
    nb_bins = int(max_diff / bin_step)
    # getting data of the histogram
    hist, gt_bins_count = np.histogram(
        np.abs(data),
        bins=nb_bins,
        range=(0, max_diff),
        density=True,
    )
    # Normalized Probability Density Function of the histogram
    pdf = hist / sum(hist)
    # Generate Cumulative Probability Function
    gt_cdf = np.cumsum(pdf)

    # Compute metric from metric class
    output_cdf, output_bins_count = metric_obj.compute_metric(data)

    np.testing.assert_allclose(output_cdf, gt_cdf, rtol=1e-02)
    np.testing.assert_allclose(output_bins_count, gt_bins_count, rtol=1e-02)


@pytest.mark.unit_tests
def test_pdf():
    """
    Test the pdf metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output, gt_bins and gt_pdf with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    # Test without percentil 98 filtering ---------------------------

    # Initialize input data
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)

    # Create metric object
    metric_obj = Metric("pdf", params={"filter_p98": "False"})

    bin_step = metric_obj._BIN_STEP
    hist, gt_bins = np.histogram(
        data[~np.isnan(data)],
        bins=np.arange(-np.nanmax(data), np.nanmax(data), bin_step),
    )

    # Normalized Probability Density Function of the histogram
    gt_pdf = hist / sum(hist)

    # Compute metric from metric class
    output_pdf, output_bins = metric_obj.compute_metric(data)

    np.testing.assert_allclose(output_pdf, gt_pdf, rtol=1e-02)
    np.testing.assert_allclose(output_bins, gt_bins, rtol=1e-02)

    # Test with percentil 98 filtering ---------------------------

    # Initialize input data
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)

    # Create metric object
    metric_obj = Metric("pdf", params={"filter_p98": "True"})

    bound = np.abs(np.nanpercentile(data, 98))
    bin_step = metric_obj._BIN_STEP
    hist, gt_bins = np.histogram(
        data[~np.isnan(data)],
        bins=np.arange(-bound, bound, bin_step),
    )

    # Normalized Probability Density Function of the histogram
    gt_pdf = hist / sum(hist)

    # Compute metric from metric class
    output_pdf, output_bins = metric_obj.compute_metric(data)

    np.testing.assert_allclose(output_pdf, gt_pdf, rtol=1e-02)
    np.testing.assert_allclose(output_bins, gt_bins, rtol=1e-02)
