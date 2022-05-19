#!/usr/bin/env python
# coding: utf8
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
This module contains functions to test Demcompare end2end with
the "gironde_test_data" test root data
"""

# strm_test_data imports
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


@pytest.mark.end2end_tests
def test_demcompare_with_gironde_test_data():
    """
    Demcompare with gironde_test_data main end2end test.
    Test that the outputs given by the Demcompare execution
    of data/gironde_test_data/input/test_config.json are
    the same as the reference ones
    in data/gironde_test_data/ref_output/

    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")

    # Load "gironde_test_data" demcompare config from input/test_config.json
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
    # Get "gironde_test_data" demcompare reference output path for
    test_ref_output_path = os.path.join(test_data_path, "ref_output")

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        # Modify test's output dir in configuration to tmp test dir
        test_cfg["output_dir"] = tmp_dir
        # Manually set the saving of internal dems to True
        test_cfg["coregistration"]["save_internal_dems"] = "True"

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "gironde_test_data_sampling_ref"
        # configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)

        # Now test demcompare output with test ref_output:

        # TEST JSON CONFIGURATION

        # Check initial config "test_config.json"
        cfg_file = "test_config.json"
        ref_output_cfg = read_config_file(
            os.path.join(test_ref_output_path, cfg_file)
        )
        output_cfg = read_config_file(os.path.join(tmp_dir, cfg_file))
        ref_output_cfg["coregistration"]["output_dir"] = tmp_dir
        ref_output_cfg["stats_opts"]["output_dir"] = tmp_dir

        np.testing.assert_equal(
            ref_output_cfg["stats_opts"]["classification_layers"]["Status"][
                "classes"
            ],
            output_cfg["stats_opts"]["classification_layers"]["Status"][
                "classes"
            ],
        )
        np.testing.assert_equal(
            ref_output_cfg["coregistration"], output_cfg["coregistration"]
        )

        # Test demcompare_results.json
        cfg_file = get_out_file_path("demcompare_results.json")
        ref_output_cfg = read_config_file(
            os.path.join(test_ref_output_path, cfg_file)
        )
        output_cfg = read_config_file(os.path.join(tmp_dir, cfg_file))
        np.testing.assert_allclose(
            ref_output_cfg["coregistration_results"]["dx"]["bias_value"],
            output_cfg["coregistration_results"]["dx"]["bias_value"],
            atol=TEST_TOL,
        )

        np.testing.assert_allclose(
            ref_output_cfg["coregistration_results"]["dy"]["bias_value"],
            output_cfg["coregistration_results"]["dy"]["bias_value"],
            atol=TEST_TOL,
        )
        np.testing.assert_allclose(
            ref_output_cfg["coregistration_results"]["dx"]["nuth_offset"],
            output_cfg["coregistration_results"]["dx"]["nuth_offset"],
            atol=TEST_TOL,
        )
        np.testing.assert_allclose(
            ref_output_cfg["coregistration_results"]["dy"]["nuth_offset"],
            output_cfg["coregistration_results"]["dy"]["nuth_offset"],
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

        # Test coreg_DEM.tif
        img = get_out_file_path("coreg_DEM.tif")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # Test reproj_coreg_DEM.tif
        img = get_out_file_path("reproj_coreg_DEM.tif")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # Test reproj_coreg_REF.tif
        img = get_out_file_path("reproj_coreg_REF.tif")
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
        np.testing.assert_allclose(ref_output_csv, output_csv, rtol=1e-2)

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


@pytest.mark.end2end_tests
def test_demcompare_with_gironde_test_data_sampling_ref():
    """
    Demcompare with classification layer with
    sampling source ref main end2end test.
    Test that the outputs given by the Demcompare execution
    of data/gironde_test_data_sampling_ref/input/test_config.json are
    the same as the reference ones
    in data/gironde_test_data_sampling_ref/ref_output/

    - Loads the data present in the gironde_test_data data directory
    - Runs demcompare
    - Tests the initial cfg file
    - Tests the demcompare_results file
    - Tests the output .tif dems
    - Tests the output .csv and .tif stats

    """
    # Get "gironde_test_data_sampling_ref" test root data
    # directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data_sampling_ref")

    # Load "gironde_test_data_sampling_ref" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    # Modify test's classification layer path to its complete path
    classif_layer_path = os.path.join(
        "input",
        test_cfg["stats_opts"]["classification_layers"]["Status"]["ref"],
    )
    test_cfg["stats_opts"]["classification_layers"]["Status"][
        "ref"
    ] = os.path.join(test_data_path, classif_layer_path)
    # Get "gironde_test_data" demcompare reference output path for
    test_ref_output_path = os.path.join(test_data_path, "ref_output")

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        # Modify test's output dir in configuration to tmp test dir
        test_cfg["output_dir"] = tmp_dir
        # Manually set the saving of internal dems to True
        test_cfg["coregistration"]["save_internal_dems"] = "True"

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "gironde_test_data_sampling_ref"
        # configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)

        # Now test demcompare output with test ref_output:

        # TEST JSON CONFIGURATION

        # Check initial config "test_config.json"
        input_cfg = "test_config.json"
        ref_output_cfg = read_config_file(
            os.path.join(test_ref_output_path, input_cfg)
        )
        ref_output_cfg["coregistration"]["output_dir"] = tmp_dir
        ref_output_cfg["stats_opts"]["output_dir"] = tmp_dir

        filled_cfg = read_config_file(os.path.join(tmp_dir, input_cfg))
        np.testing.assert_equal(
            ref_output_cfg["stats_opts"]["classification_layers"]["Status"][
                "classes"
            ],
            filled_cfg["stats_opts"]["classification_layers"]["Status"][
                "classes"
            ],
        )
        np.testing.assert_equal(
            ref_output_cfg["coregistration"], filled_cfg["coregistration"]
        )

        # Test demcompare_results.json
        demcompare_results_path = get_out_file_path("demcompare_results.json")
        ref_demcompare_results = read_config_file(
            os.path.join(test_ref_output_path, demcompare_results_path)
        )
        demcompare_results = read_config_file(
            os.path.join(tmp_dir, demcompare_results_path)
        )
        np.testing.assert_allclose(
            ref_demcompare_results["coregistration_results"]["dx"][
                "bias_value"
            ],
            demcompare_results["coregistration_results"]["dx"]["bias_value"],
            atol=TEST_TOL,
        )

        np.testing.assert_allclose(
            ref_demcompare_results["coregistration_results"]["dy"][
                "bias_value"
            ],
            demcompare_results["coregistration_results"]["dy"]["bias_value"],
            atol=TEST_TOL,
        )
        np.testing.assert_allclose(
            ref_demcompare_results["coregistration_results"]["dx"][
                "nuth_offset"
            ],
            demcompare_results["coregistration_results"]["dx"]["nuth_offset"],
            atol=TEST_TOL,
        )
        np.testing.assert_allclose(
            ref_demcompare_results["coregistration_results"]["dy"][
                "nuth_offset"
            ],
            demcompare_results["coregistration_results"]["dy"]["nuth_offset"],
            atol=TEST_TOL,
        )
        np.testing.assert_allclose(
            ref_demcompare_results["alti_results"]["dz"]["bias_value"],
            demcompare_results["alti_results"]["dz"]["bias_value"],
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

        # Test coreg_DEM.tif
        img = get_out_file_path("coreg_DEM.tif")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # Test reproj_coreg_DEM.tif
        img = get_out_file_path("reproj_coreg_DEM.tif")
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # Test reproj_coreg_REF.tif
        img = get_out_file_path("reproj_coreg_REF.tif")
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

        # Test ref_support_map_rectif.tif
        img = "stats/Status/ref_support_map_rectif.tif"
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)

        # TEST FUSION_LAYER STATS

        # Test stats/fusion_layer/stats_results_standard.csv
        file = "stats/fusion_layer/stats_results_standard.csv"
        ref_output_csv = read_csv_file(os.path.join(test_ref_output_path, file))
        output_csv = read_csv_file(os.path.join(tmp_dir, file))
        np.testing.assert_allclose(ref_output_csv, output_csv, atol=TEST_TOL)

        # Test ref_fusion_layer.tif
        img = "stats/fusion_layer/ref_fusion_layer.tif"
        ref_output_data = os.path.join(test_ref_output_path, img)
        output_data = os.path.join(tmp_dir, img)
        assert_same_images(ref_output_data, output_data, atol=TEST_TOL)
