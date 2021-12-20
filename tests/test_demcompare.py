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

import json
import os
import unittest
from typing import Dict

import numpy as np
import rasterio

import demcompare


class TestPandora(unittest.TestCase):
    """
    TestPandora class allows to test the pandora pipeline
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        # Baseline directory with ground truths
        self.baseline_dir = os.path.join(os.getcwd(), "tests/test_baseline")
        # Output test directory with ground truths
        self.output_test_dir = os.path.join(os.getcwd(), "tests/test_output")
        # Set current directory on tests for execution
        os.chdir(os.path.join(os.getcwd(), "tests"))
        # Execute Demcompare on test_config.json
        test_cfg = "test_config.json"
        demcompare.run(test_cfg)

    @staticmethod
    def read_config_file(config_file: str) -> Dict[str, dict]:
        """
        Read a json configuration file

        :param config_file: path to a json file
        :type config_file: string
        :return user_cfg: configuration dictionary
        :rtype: dict
        """
        with open(
            config_file, "r"  # pylint: disable=bad-option-value
        ) as file_:
            user_cfg = json.load(file_)
        return user_cfg

    @staticmethod
    def read_csv_file(csv_file: str):
        """
        Read a csv file and save its number values to float

        :param csv_file: path to a csv file
        :type csv_file: string
        """
        lines = open(csv_file, "r").readlines()
        output_file = []

        for idx, line in enumerate(lines):
            # Obtain colums
            cols = line.split(",")
            # Last column ends with \n
            cols[-1] = cols[-1].split("\n")[0]
            # First line are titles
            if idx == 0:
                continue
            # If it is the stats csv, do not convert to float first col
            if len(cols) > 2:
                output_file.append(np.array(cols[1:], dtype=float))
                continue
            # Convert to float
            output_file.append(np.array(cols, dtype=float))
        return output_file

    def test_demcompare(self):
        """
        Test that all the outputs given by the Demcompare execution
        on test_output are the same as the ones on test_baseline

        """
        # Test initial_dh.tif
        img = "/initial_dh.tif"
        baseline_data = rasterio.open(self.baseline_dir + img).read(1)
        output_data = rasterio.open(self.output_test_dir + img).read(1)
        np.testing.assert_allclose(baseline_data, output_data, atol=1e-05)

        # Test final_dh.tif
        img = "/final_dh.tif"
        baseline_data = rasterio.open(self.baseline_dir + img).read(1)
        output_data = rasterio.open(self.output_test_dir + img).read(1)
        np.testing.assert_allclose(baseline_data, output_data, atol=1e-05)

        # Test initial_dem_diff_pdf.png
        img = "/initial_dem_diff_pdf.png"
        baseline_data = rasterio.open(self.baseline_dir + img).read(1)
        output_data = rasterio.open(self.output_test_dir + img).read(1)
        np.testing.assert_allclose(baseline_data, output_data, atol=1e-05)

        # Test initial_dem_diff_pdf.png
        img = "/initial_dem_diff_pdf.png"
        baseline_data = rasterio.open(self.baseline_dir + img).read(1)
        output_data = rasterio.open(self.output_test_dir + img).read(1)
        np.testing.assert_allclose(baseline_data, output_data, atol=1e-05)

        # Test final_dem_diff_pdf.png
        img = "/final_dem_diff_pdf.png"
        baseline_data = rasterio.open(self.baseline_dir + img).read(1)
        output_data = rasterio.open(self.output_test_dir + img).read(1)
        np.testing.assert_allclose(baseline_data, output_data, atol=1e-05)

        # Test final_config.json
        cfg = "/final_config.json"
        baseline_cfg = self.read_config_file(self.baseline_dir + cfg)
        output_cfg = self.read_config_file(self.output_test_dir + cfg)
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
        baseline_cfg = self.read_config_file(self.baseline_dir + cfg)
        output_cfg = self.read_config_file(self.output_test_dir + cfg)
        np.testing.assert_equal(
            baseline_cfg["stats_opts"], output_cfg["stats_opts"]
        )
        np.testing.assert_equal(
            baseline_cfg["plani_opts"], output_cfg["plani_opts"]
        )

        # Test final_dem_diff_pdf.csv
        file = "/final_dem_diff_pdf.csv"
        baseline_csv = self.read_csv_file(self.baseline_dir + file)
        output_csv = self.read_csv_file(self.output_test_dir + file)
        np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)

        # Test initial_dem_diff_pdf.csv
        file = "/initial_dem_diff_pdf.csv"
        baseline_csv = self.read_csv_file(self.baseline_dir + file)
        output_csv = self.read_csv_file(self.output_test_dir + file)
        np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)

        # Test stats/slope/stats_results_standard.csv
        file = "/stats/slope/stats_results_standard.csv"
        baseline_csv = self.read_csv_file(self.baseline_dir + file)
        output_csv = self.read_csv_file(self.output_test_dir + file)
        np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)

        # Test stats/slope/stats_results_incoherent-classification.csv
        file = "/stats/slope/stats_results_incoherent-classification.csv"
        baseline_csv = self.read_csv_file(self.baseline_dir + file)
        output_csv = self.read_csv_file(self.output_test_dir + file)
        np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)

        # Test stats/slope/stats_results_coherent-classification.csv
        file = "/stats/slope/stats_results_coherent-classification.csv"
        baseline_csv = self.read_csv_file(self.baseline_dir + file)
        output_csv = self.read_csv_file(self.output_test_dir + file)
        np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)

        # Test snapshots/final_dem_diff_cdf.csv
        file = "/snapshots/final_dem_diff_cdf.csv"
        baseline_csv = self.read_csv_file(self.baseline_dir + file)
        output_csv = self.read_csv_file(self.output_test_dir + file)
        np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)

        # Test snapshots/initial_dem_diff_cdf.csv
        file = "/snapshots/initial_dem_diff_cdf.csv"
        baseline_csv = self.read_csv_file(self.baseline_dir + file)
        output_csv = self.read_csv_file(self.output_test_dir + file)
        np.testing.assert_allclose(baseline_csv, output_csv, atol=1e-05)
