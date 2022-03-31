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

# Standard imports
import os
from tempfile import TemporaryDirectory

# Third party imports
import numpy as np
import pytest

# Demcompare imports
import demcompare
from demcompare.initialization import read_config_file, save_config_file
from demcompare.output_tree_design import get_out_file_path

# Tests helpers
from .helpers import (
    TEST_TOL,
    assert_same_images,
    demcompare_test_data_path,
    read_csv_file,
    temporary_dir,
)

# Demcompare imports


@pytest.mark.end2end_tests
def test_demcompare_standard_outputs():
    """
    Standard main end2end test.
    Test that the outputs given by the Demcompare execution
    of data/standard/input/test_config.json are the same as the reference ones
    in data/standard/ref_output/

    """
    # Get "standard" test root data directory absolute path
    test_data_path = demcompare_test_data_path("standard")

    # Load "standard" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    # Get "standard" demcompare reference output path for
    test_ref_output_path = os.path.join(test_data_path, "ref_output")

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:

        # Modify test's output dir in configuration to tmp test dir
        test_cfg["outputDir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "standard" configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)

        # Now test demcompare output with test ref_output:

        # TEST JSON CONFIGURATION

        # Check initial config "test_config.json"
        cfg_file = "test_config.json"
        ref_output_cfg = read_config_file(
            os.path.join(test_ref_output_path, cfg_file)
        )
        output_cfg = read_config_file(os.path.join(tmp_dir, cfg_file))
        np.testing.assert_equal(
            ref_output_cfg["stats_opts"], output_cfg["stats_opts"]
        )
        np.testing.assert_equal(
            ref_output_cfg["plani_opts"], output_cfg["plani_opts"]
        )

        # Test final_config.json
        cfg_file = get_out_file_path("final_config.json")
        ref_output_cfg = read_config_file(
            os.path.join(test_ref_output_path, cfg_file)
        )
        output_cfg = read_config_file(os.path.join(tmp_dir, cfg_file))
        np.testing.assert_allclose(
            ref_output_cfg["plani_results"]["dx"]["bias_value"],
            output_cfg["plani_results"]["dx"]["bias_value"],
            atol=TEST_TOL,
        )

        np.testing.assert_allclose(
            ref_output_cfg["plani_results"]["dy"]["bias_value"],
            output_cfg["plani_results"]["dy"]["bias_value"],
            atol=TEST_TOL,
        )

        np.testing.assert_allclose(
            ref_output_cfg["plani_results"]["gdal_translate_bounds"]["lry"],
            output_cfg["plani_results"]["gdal_translate_bounds"]["lry"],
            atol=TEST_TOL,
        )

        np.testing.assert_allclose(
            ref_output_cfg["plani_results"]["gdal_translate_bounds"]["lrx"],
            output_cfg["plani_results"]["gdal_translate_bounds"]["lrx"],
            atol=TEST_TOL,
        )

        np.testing.assert_allclose(
            ref_output_cfg["plani_results"]["gdal_translate_bounds"]["uly"],
            output_cfg["plani_results"]["gdal_translate_bounds"]["uly"],
            atol=TEST_TOL,
        )

        np.testing.assert_allclose(
            ref_output_cfg["plani_results"]["gdal_translate_bounds"]["ulx"],
            output_cfg["plani_results"]["gdal_translate_bounds"]["ulx"],
            atol=TEST_TOL,
        )

        np.testing.assert_allclose(
            ref_output_cfg["alti_results"]["dz"]["bias_value"],
            output_cfg["alti_results"]["dz"]["bias_value"],
            atol=TEST_TOL,
        )
        # TEST DIFF TIF

        # Test initial_dh.tif
        img = get_out_file_path("initial_dh.tif")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # Test final_dh.tif
        img = get_out_file_path("final_dh.tif")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # TEST PNG SNAPSHOTS

        # Test initial_dem_diff_pdf.png
        img = get_out_file_path("initial_dem_diff_pdf.png")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # Test final_dem_diff_pdf.png
        img = get_out_file_path("final_dem_diff_pdf.png")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # Test snapshots/initial_dem_diff_cdf.png
        img = get_out_file_path("initial_dem_diff_cdf.png")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # Test snapshots/initial_dem_diff_cdf.png
        img = get_out_file_path("final_dem_diff_cdf.png")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # TESTS CSV SNAPSHOTS

        # Test initial_dem_diff_pdf.csv
        file = get_out_file_path("initial_dem_diff_pdf.csv")
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test final_dem_diff_pdf.csv
        file = get_out_file_path("final_dem_diff_pdf.csv")
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test snapshots/initial_dem_diff_cdf.csv
        file = get_out_file_path("initial_dem_diff_cdf.csv")
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test snapshots/final_dem_diff_cdf.csv
        file = get_out_file_path("final_dem_diff_cdf.csv")
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # TEST CSV STATS

        # Test stats/slope/stats_results_standard.csv
        file = "stats/slope/stats_results_standard.csv"
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test stats/slope/stats_results_incoherent-classification.csv
        file = "stats/slope/stats_results_incoherent-classification.csv"
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test stats/slope/stats_results_coherent-classification.csv
        file = "stats/slope/stats_results_coherent-classification.csv"
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)


@pytest.mark.end2end_tests
def test_demcompare_standard_outputs_with_roi():
    """
    Standard main end2end test with ROI input.
    Test that the outputs given by the Demcompare execution
    of data/standard_roi/input/test_config.json are the same
    as the reference ones in data/standard_roi/ref_output/

    """
    # Get "standard_roi" test root data directory absolute path
    test_data_path = demcompare_test_data_path("standard_roi")

    # Load "standard_roi" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    # Get "standard_roi" demcompare reference output path for
    test_ref_output_path = os.path.join(test_data_path, "ref_output")

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:

        # Modify test's output dir in configuration to tmp test dir
        test_cfg["outputDir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "standard" configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)

        # Now test demcompare output with test ref_output:

        # TEST JSON CONFIGURATION

        # Check initial config "test_config.json"
        cfg_file = "test_config.json"
        ref_output_cfg = read_config_file(
            os.path.join(test_ref_output_path, cfg_file)
        )
        output_cfg = read_config_file(os.path.join(tmp_dir, cfg_file))
        np.testing.assert_equal(
            ref_output_cfg["stats_opts"], output_cfg["stats_opts"]
        )
        np.testing.assert_equal(
            ref_output_cfg["plani_opts"], output_cfg["plani_opts"]
        )

        # Test final_config.json
        cfg_file = get_out_file_path("final_config.json")
        ref_output_cfg = read_config_file(
            os.path.join(test_ref_output_path, cfg_file)
        )
        output_cfg = read_config_file(os.path.join(tmp_dir, cfg_file))
        np.testing.assert_allclose(
            ref_output_cfg["plani_results"]["dx"]["bias_value"],
            output_cfg["plani_results"]["dx"]["bias_value"],
            atol=TEST_TOL,
        )

        np.testing.assert_allclose(
            ref_output_cfg["plani_results"]["dy"]["bias_value"],
            output_cfg["plani_results"]["dy"]["bias_value"],
            atol=TEST_TOL,
        )

        np.testing.assert_allclose(
            ref_output_cfg["alti_results"]["dz"]["bias_value"],
            output_cfg["alti_results"]["dz"]["bias_value"],
            atol=TEST_TOL,
        )

        # TEST DIFF TIF

        # Test initial_dh.tif
        img = get_out_file_path("initial_dh.tif")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # Test final_dh.tif
        img = get_out_file_path("final_dh.tif")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # TESTS CSV SNAPSHOTS

        # Test initial_dem_diff_pdf.csv
        file = get_out_file_path("initial_dem_diff_pdf.csv")
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test final_dem_diff_pdf.csv
        file = get_out_file_path("final_dem_diff_pdf.csv")
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test snapshots/initial_dem_diff_cdf.csv
        file = get_out_file_path("initial_dem_diff_cdf.csv")
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test snapshots/final_dem_diff_cdf.csv
        file = get_out_file_path("final_dem_diff_cdf.csv")
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # TEST CSV STATS

        # Test stats/slope/stats_results_standard.csv
        file = "stats/slope/stats_results_standard.csv"
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test stats/slope/stats_results_coherent-classification.csv
        file = "stats/slope/stats_results_coherent-classification.csv"
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)


@pytest.mark.end2end_tests
def test_demcompare_with_classification_layer():
    """
    Demcompare with classification layer main end2end test.
    Test that the outputs given by the Demcompare execution
    of data/classification_layer/input/test_config.json are
    the same as the reference ones
    in data/classification_layer/ref_output/

    """
    # Get "classification_layer" test root data directory absolute path
    test_data_path = demcompare_test_data_path("classification_layer")

    # Load "classification_layer" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    # Modify test's classification layer path to its complete path
    classif_layer_path = os.path.join(
        "input",
        test_cfg["stats_opts"]["classification_layers"]["Status"]["dsm"],
    )
    test_cfg["stats_opts"]["classification_layers"]["Status"][
        "dsm"
    ] = os.path.join(test_data_path, classif_layer_path)
    # Get "classification_layer" demcompare reference output path for
    test_ref_output_path = os.path.join(test_data_path, "ref_output")

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:

        # Modify test's output dir in configuration to tmp test dir
        test_cfg["outputDir"] = tmp_dir
        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "standard" configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)

        # Now test demcompare output with test ref_output:

        # TEST JSON CONFIGURATION

        # Check initial config "test_config.json"
        cfg_file = "test_config.json"
        ref_output_cfg = read_config_file(
            os.path.join(test_ref_output_path, cfg_file)
        )
        output_cfg = read_config_file(os.path.join(tmp_dir, cfg_file))
        np.testing.assert_equal(
            ref_output_cfg["stats_opts"]["classification_layers"]["Status"][
                "classes"
            ],
            output_cfg["stats_opts"]["classification_layers"]["Status"][
                "classes"
            ],
        )
        np.testing.assert_equal(
            ref_output_cfg["plani_opts"], output_cfg["plani_opts"]
        )

        # Test final_config.json
        cfg_file = get_out_file_path("final_config.json")
        ref_output_cfg = read_config_file(
            os.path.join(test_ref_output_path, cfg_file)
        )
        output_cfg = read_config_file(os.path.join(tmp_dir, cfg_file))
        np.testing.assert_allclose(
            ref_output_cfg["plani_results"]["dx"]["bias_value"],
            output_cfg["plani_results"]["dx"]["bias_value"],
            atol=TEST_TOL,
        )

        np.testing.assert_allclose(
            ref_output_cfg["plani_results"]["dy"]["bias_value"],
            output_cfg["plani_results"]["dy"]["bias_value"],
            atol=TEST_TOL,
        )

        np.testing.assert_allclose(
            ref_output_cfg["alti_results"]["dz"]["bias_value"],
            output_cfg["alti_results"]["dz"]["bias_value"],
            atol=TEST_TOL,
        )

        # TEST DIFF TIF

        # Test initial_dh.tif
        img = get_out_file_path("initial_dh.tif")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # Test final_dh.tif
        img = get_out_file_path("final_dh.tif")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # TEST SLOPE STATS

        # Test stats/slope/stats_results_standard.csv
        file = "stats/slope/stats_results_standard.csv"
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test stats/slope/stats_results_coherent-classification.csv
        file = "stats/slope/stats_results_coherent-classification.csv"
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test dsm_support_map_rectif.tif
        img = "stats/slope/dsm_support_map_rectif.tif"
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # Test ref_support_map_rectif.tif
        img = "stats/slope/ref_support_map_rectif.tif"
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # TEST STATUS CLASSIFICATION LAYER STATS

        # Test stats/Status/stats_results_standard.csv
        file = "stats/Status/stats_results_standard.csv"
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test dsm_support_map_rectif.tif
        img = "stats/Status/dsm_support_map_rectif.tif"
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # TEST FUSION_LAYER STATS

        # Test stats/fusion_layer/stats_results_standard.csv
        file = "stats/fusion_layer/stats_results_standard.csv"
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test dsm_fusion_layer.tif
        img = "stats/fusion_layer/dsm_fusion_layer.tif"
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)
