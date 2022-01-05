#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions to test Demcompare.
"""

import os

import numpy as np
import pytest

from .helpers import (
    assert_same_images,
    demcompare_gt_data_path,
    demcompare_output_data_path,
    read_config_file,
    read_csv_file,
    run_demcompare,
)


@pytest.mark.end2end_tests
def test_demcompare_outputs():
    """
    Test that all the outputs given by the Demcompare execution
    on test_output are the same as the ones on test_baseline

    """

    # Baseline directory with ground truths
    baseline_dir = demcompare_gt_data_path()
    # Output test directory with ground truths
    output_test_dir = demcompare_output_data_path()
    # Tests config name
    test_cfg = "test_config.json"
    # Run demcompare
    run_demcompare(test_cfg)

    # Test initial_dh.tif
    img = "/initial_dh.tif"
    baseline_data = os.path.join(baseline_dir + img)
    output_data = os.path.join(output_test_dir + img)
    assert_same_images(baseline_data, output_data, atol=1e-05)

    # Test final_dh.tif
    img = "/final_dh.tif"
    baseline_data = os.path.join(baseline_dir + img)
    output_data = os.path.join(output_test_dir + img)
    assert_same_images(baseline_data, output_data, atol=1e-05)

    # Test initial_dem_diff_pdf.png
    img = "/initial_dem_diff_pdf.png"
    baseline_data = os.path.join(baseline_dir + img)
    output_data = os.path.join(output_test_dir + img)
    assert_same_images(baseline_data, output_data, atol=1e-05)

    # Test initial_dem_diff_pdf.png
    img = "/initial_dem_diff_pdf.png"
    baseline_data = os.path.join(baseline_dir + img)
    output_data = os.path.join(output_test_dir + img)
    assert_same_images(baseline_data, output_data, atol=1e-05)

    # Test final_dem_diff_pdf.png
    img = "/final_dem_diff_pdf.png"
    baseline_data = os.path.join(baseline_dir + img)
    output_data = os.path.join(output_test_dir + img)
    assert_same_images(baseline_data, output_data, atol=1e-05)

    # Test final_config.json
    cfg = "/final_config.json"
    baseline_cfg = read_config_file(baseline_dir + cfg)
    output_cfg = read_config_file(output_test_dir + cfg)
    np.testing.assert_allclose(
        baseline_cfg["plani_results"]["dx"]["bias_value"],
        output_cfg["plani_results"]["dx"]["bias_value"],
    )
    np.testing.assert_allclose(
        baseline_cfg["plani_results"]["dy"]["bias_value"],
        output_cfg["plani_results"]["dy"]["bias_value"],
    )
    np.testing.assert_allclose(
        baseline_cfg["alti_results"]["dz"]["bias_value"],
        output_cfg["alti_results"]["dz"]["bias_value"],
    )

    # Test test_config.json
    cfg = "/test_config.json"
    baseline_cfg = read_config_file(baseline_dir + cfg)
    output_cfg = read_config_file(output_test_dir + cfg)
    np.testing.assert_equal(
        baseline_cfg["stats_opts"], output_cfg["stats_opts"]
    )
    np.testing.assert_equal(
        baseline_cfg["plani_opts"], output_cfg["plani_opts"]
    )

    # Test final_dem_diff_pdf.csv
    file = "/final_dem_diff_pdf.csv"
    baseline_csv = read_csv_file(baseline_dir + file)
    output_csv = read_csv_file(output_test_dir + file)
    np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)

    # Test initial_dem_diff_pdf.csv
    file = "/initial_dem_diff_pdf.csv"
    baseline_csv = read_csv_file(baseline_dir + file)
    output_csv = read_csv_file(output_test_dir + file)
    np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)

    # Test stats/slope/stats_results_standard.csv
    file = "/stats/slope/stats_results_standard.csv"
    baseline_csv = read_csv_file(baseline_dir + file)
    output_csv = read_csv_file(output_test_dir + file)
    np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)

    # Test stats/slope/stats_results_incoherent-classification.csv
    file = "/stats/slope/stats_results_incoherent-classification.csv"
    baseline_csv = read_csv_file(baseline_dir + file)
    output_csv = read_csv_file(output_test_dir + file)
    np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)

    # Test stats/slope/stats_results_coherent-classification.csv
    file = "/stats/slope/stats_results_coherent-classification.csv"
    baseline_csv = read_csv_file(baseline_dir + file)
    output_csv = read_csv_file(output_test_dir + file)
    np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)

    # Test snapshots/final_dem_diff_cdf.csv
    file = "/snapshots/final_dem_diff_cdf.csv"
    baseline_csv = read_csv_file(baseline_dir + file)
    output_csv = read_csv_file(output_test_dir + file)
    np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)

    # Test snapshots/initial_dem_diff_cdf.csv
    file = "/snapshots/initial_dem_diff_cdf.csv"
    baseline_csv = read_csv_file(baseline_dir + file)
    output_csv = read_csv_file(output_test_dir + file)
    np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)
