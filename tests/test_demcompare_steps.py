#!/usr/bin/env python
# coding: utf8
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
# pylint:disable=duplicate-code,too-many-lines
"""
This module contains functions to test Demcompare coregistration
and statistics steps independently with
the "gironde_test_data" test root data
"""

# Standard imports
import os
from tempfile import TemporaryDirectory

# Third party imports
import numpy as np
import pytest

# Demcompare imports
import demcompare
from demcompare.helpers_init import (
    compute_initialization,
    read_config_file,
    save_config_file,
)

# Tests helpers
from .helpers import demcompare_test_data_path, temporary_dir


@pytest.mark.end2end_tests
@pytest.mark.functional_tests
def test_demcompare_coregistration_step_with_gironde_test_data():
    """
    Demcompare with only coregistration step end2end test.
    Input data:
    - Input dems and configuration present in the
      "gironde_test_data/input" test data directory
    Validation data:
    - Output data present in the
      "gironde_test_data/ref_output" test data directory
    Validation process:
    - Reads the input configuration file
    - Deletes the statistics step of the configuration file
    - Runs demcompare on a temporary directory
    - Checks that the output files are the same as ground truth
    - Checked files: coregistration_results.json
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")

    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    # Since we only want the coregistration step to be run,
    # Pop the statistics step of the cfg
    test_cfg.pop("statistics")
    test_cfg["input_sec"].pop("classification_layers")

    # Input configuration is
    # "input_ref": {
    #     "path": "./Gironde.tif",
    #     "zunit": "m"
    # },
    # "input_sec": {
    #     "path": "./FinalWaveBathymetry_T3..._invert.TIF",
    #     "zunit": "m",
    #     "nodata": -32768,
    #     },
    # "coregistration": {
    #     "method_name": "nuth_kaab_internal",
    #     "number_of_iterations": 6,
    #     "estimated_initial_shift_x": 0,
    #     "estimated_initial_shift_y": 0
    # }

    # Get "gironde_test_data" demcompare reference output path for
    test_ref_output_path = os.path.join(test_data_path, "ref_output")

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        # Modify test's output dir in configuration to tmp test dir
        test_cfg["output_dir"] = tmp_dir
        # Manually set the saving of internal dems to True
        test_cfg["coregistration"]["save_optional_outputs"] = True

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "gironde_test_data"
        # configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)

        # Now test demcompare output with test ref_output:

        # Test coregistration_results.json
        cfg_file = "./coregistration/coregistration_results.json"
        ref_coregistration_results = read_config_file(
            os.path.join(test_ref_output_path, cfg_file)
        )
        coregistration_results = read_config_file(
            os.path.join(tmp_dir, cfg_file)
        )
        np.testing.assert_equal(
            ref_coregistration_results["coregistration_results"]["dx"][
                "total_bias_value"
            ],
            coregistration_results["coregistration_results"]["dx"][
                "total_bias_value"
            ],
        )
        np.testing.assert_equal(
            ref_coregistration_results["coregistration_results"]["dy"][
                "total_bias_value"
            ],
            coregistration_results["coregistration_results"]["dy"][
                "total_bias_value"
            ],
        )
        np.testing.assert_equal(
            ref_coregistration_results["coregistration_results"]["dx"][
                "nuth_offset"
            ],
            coregistration_results["coregistration_results"]["dx"][
                "nuth_offset"
            ],
        )
        np.testing.assert_equal(
            ref_coregistration_results["coregistration_results"]["dy"][
                "nuth_offset"
            ],
            coregistration_results["coregistration_results"]["dy"][
                "nuth_offset"
            ],
        )
        np.testing.assert_equal(
            ref_coregistration_results["coregistration_results"]["dx"][
                "total_offset"
            ],
            coregistration_results["coregistration_results"]["dx"][
                "total_offset"
            ],
        )
        np.testing.assert_equal(
            ref_coregistration_results["coregistration_results"]["dy"][
                "total_offset"
            ],
            coregistration_results["coregistration_results"]["dy"][
                "total_offset"
            ],
        )
        np.testing.assert_equal(
            ref_coregistration_results["coregistration_results"]["dz"][
                "total_bias_value"
            ],
            coregistration_results["coregistration_results"]["dz"][
                "total_bias_value"
            ],
        )


@pytest.mark.end2end_tests
@pytest.mark.functional_tests
def test_demcompare_statistics_step_with_gironde_test_data():
    """
    Demcompare with only statistics step on two input dems
    end2end test.
    Input data:
    - Input dems and configuration present in the
      "gironde_test_data/input" test data directory
    Validation process:
    - Reads the input configuration file
    - Deletes the coregistration step of the configuration file
    - Runs demcompare on a temporary directory
    - Checks that no error is raised
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")

    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    # Since we only want the statistics step to be run,
    # Pop the coregistration step of the cfg
    test_cfg.pop("coregistration")

    # Input configuration is
    # "input_ref": {
    #     "path": "./Gironde.tif",
    #     "zunit": "m"
    # },
    # "input_sec": {
    #     "path": "./FinalWaveBathymetry_T3..._invert.TIF",
    #     "zunit": "m",
    #     "nodata": -32768,
    #     "classification_layers": {
    #         "Status": {
    #             "map_path": "./FinalWaveBathymetry_T3...TIF"}
    # }},
    # "statistics": {
    #     "remove_outliers": false,
    #     "classification_layers": {
    #         "Status": {
    #             "type": "segmentation",
    #            "classes": {"valid": [0],
    #                       "KO": [1],
    #                       "Land": [2],
    #                       "NoData": [3],
    #                       "Outside_detector": [4]}
    #         },
    #         "Slope0": {
    #              "type": "slope",
    #             "ranges": [0, 10, 25, 50, 90]
    #         },
    #         "Fusion0": {
    #             "type": "fusion",
    #             "sec": ["Slope0", "Status"]
    #         }
    #     }
    # }

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        # Modify test's output dir in configuration to tmp test dir
        test_cfg["output_dir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "gironde_test_data"
        # configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)


@pytest.mark.end2end_tests
@pytest.mark.functional_tests
def test_demcompare_statistics_step_input_ref_with_gironde_test_data():
    """
    Demcompare with only statistics step on one input dem
    end2end test.
    Input data:
    - Input dems and configuration present in the
      "gironde_test_data_sampling_ref/input" test data directory
    Validation process:
    - Reads the input configuration file
    - Deletes the input_sec of the configuration file
    - Deletes the coregistration step of the configuration file
    - Runs demcompare on a temporary directory
    - Checks that no error is raised
    """
    # Get "gironde_test_data" test root
    # data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data_sampling_ref")

    # Load "gironde_test_data_sampling_ref"
    # demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    # Since we only want the statistics step to be run,
    # Pop the coregistration step of the cfg
    test_cfg.pop("coregistration")
    # Since we only want the input_ref, pop the input_sec
    # of the cfg
    test_cfg.pop("input_sec")
    test_cfg["statistics"]["alti-diff"]["classification_layers"].pop("Fusion0")
    test_cfg["statistics"]["ref"] = test_cfg["statistics"].pop("alti-diff")

    # Input configuration is
    # "input_ref": {
    #     "path": "./Gironde.tif",
    #     "zunit": "m",
    #     "classification_layers": {
    #        "Status": {
    #            "map_path": "./ref_status.tif"
    #        }
    # },
    # "input_sec": {
    #     "path": "./FinalWaveBathymetry_....TIF",
    #     "zunit": "m",
    #     "nodata": -32768,
    #     },
    # "statistics": {
    #     "remove_outliers": false,
    #     "classification_layers": {
    #         "Status": {
    #             "type": "segmentation",
    #            "classes": {"valid": [0],
    #                       "KO": [1],
    #                       "Land": [2],
    #                       "NoData": [3],
    #                       "Outside_detector": [4]}
    #         },
    #         "Slope0": {
    #              "type": "slope",
    #             "ranges": [0, 10, 25, 50, 90]
    #         }
    #     }
    # }

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        # Modify test's output dir in configuration to tmp test dir
        test_cfg["output_dir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "gironde_test_data"
        # configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)


@pytest.mark.end2end_tests
@pytest.mark.functional_tests
def test_demcompare_statistics_step_input_sec_with_gironde_test_data():
    """
    Demcompare with only statistics step on one input dem
    end2end test.
    Input data:
    - Input dems and configuration present in the
      "gironde_test_data_sampling_sec/input" test data directory
    Validation process:
    - Reads the input configuration file
    - Deletes the coregistration step of the configuration file
    - Runs demcompare on a temporary directory
    - Checks that no error is raised
    """
    # Get "gironde_test_data" test root
    # data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data_sampling_ref")

    # Load "gironde_test_data_sampling_sec"
    # demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    # Since we only want the statistics step to be run,
    # Pop the coregistration step of the cfg
    test_cfg["statistics"]["alti-diff"]["classification_layers"].pop("Fusion0")
    test_cfg["statistics"]["sec"] = test_cfg["statistics"].pop("alti-diff")

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        # Modify test's output dir in configuration to tmp test dir
        test_cfg["output_dir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "gironde_test_data"
        # configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)


@pytest.mark.end2end_tests
@pytest.mark.functional_tests
def test_demcompare_statistics_step_angular_diff_with_gironde_test_data():
    """
    Demcompare with angular difference config
    Input data:
    - Input dems and configuration present in the
      "gironde_test_data_sampling_ref/input" test data directory
    Validation process:
    - Reads the input configuration file
    - Deletes the input_sec of the configuration file
    - Deletes the coregistration step of the configuration file
    - Runs demcompare on a temporary directory
    - Checks that no error is raised
    """
    # Get "gironde_test_data" test root
    # data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data_sampling_ref")

    # Load "gironde_test_data_sampling_ref"
    # demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    test_cfg["statistics"]["angular-diff"] = test_cfg["statistics"].pop(
        "alti-diff"
    )

    # Input configuration is
    # "input_ref": {
    #     "path": "./Gironde.tif",
    #     "zunit": "m",
    #     "classification_layers": {
    #        "Status": {
    #            "map_path": "./ref_status.tif"
    #        }
    # },
    # "input_sec": {
    #     "path": "./FinalWaveBathymetry_....TIF",
    #     "zunit": "m",
    #     "nodata": -32768,
    #     },
    # "statistics": {
    #     "remove_outliers": false,
    #     "classification_layers": {
    #         "Status": {
    #             "type": "segmentation",
    #            "classes": {"valid": [0],
    #                       "KO": [1],
    #                       "Land": [2],
    #                       "NoData": [3],
    #                       "Outside_detector": [4]}
    #         },
    #         "Slope0": {
    #              "type": "slope",
    #             "ranges": [0, 10, 25, 50, 90]
    #         }
    #     }
    # }

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        # Modify test's output dir in configuration to tmp test dir
        test_cfg["output_dir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "gironde_test_data"
        # configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)


@pytest.mark.end2end_tests
@pytest.mark.functional_tests
def test_demcompare_statistics_step_alti_diff_norm_diff_with_gironde_test_data():  # noqa: E501, B950 # pylint: disable=line-too-long
    """
    Demcompare with altitude difference normalized by the slope config
    Input data:
    - Input dems and configuration present in the
      "gironde_test_data_sampling_ref/input" test data directory
    Validation process:
    - Reads the input configuration file
    - Deletes the input_sec of the configuration file
    - Deletes the coregistration step of the configuration file
    - Runs demcompare on a temporary directory
    - Checks that no error is raised
    """
    # Get "gironde_test_data" test root
    # data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data_sampling_ref")

    # Load "gironde_test_data_sampling_ref"
    # demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    test_cfg["statistics"]["alti-diff-slope-norm"] = test_cfg["statistics"].pop(
        "alti-diff"
    )

    # Input configuration is
    # "input_ref": {
    #     "path": "./Gironde.tif",
    #     "zunit": "m",
    #     "classification_layers": {
    #        "Status": {
    #            "map_path": "./ref_status.tif"
    #        }
    # },
    # "input_sec": {
    #     "path": "./FinalWaveBathymetry_....TIF",
    #     "zunit": "m",
    #     "nodata": -32768,
    #     },
    # "statistics": {
    #     "remove_outliers": false,
    #     "classification_layers": {
    #         "Status": {
    #             "type": "segmentation",
    #            "classes": {"valid": [0],
    #                       "KO": [1],
    #                       "Land": [2],
    #                       "NoData": [3],
    #                       "Outside_detector": [4]}
    #         },
    #         "Slope0": {
    #              "type": "slope",
    #             "ranges": [0, 10, 25, 50, 90]
    #         }
    #     }
    # }

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        # Modify test's output dir in configuration to tmp test dir
        test_cfg["output_dir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "gironde_test_data"
        # configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)


@pytest.mark.end2end_tests
@pytest.mark.functional_tests
def test_demcompare_statistics_step_curvature_input_ref_with_gironde_test_data():  # noqa: E501, B950 # pylint: disable=line-too-long
    """
    Demcompare with only statistics step on one input dem
    end2end test.
    Input data:
    - Input dems and configuration present in the
      "gironde_test_data_sampling_ref/input" test data directory
    Validation process:
    - Reads the input configuration file
    - Deletes the input_sec of the configuration file
    - Deletes the coregistration step of the configuration file
    - Runs demcompare on a temporary directory
    - Checks that no error is raised
    """
    # Get "gironde_test_data" test root
    # data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data_sampling_ref")

    # Load "gironde_test_data_sampling_ref"
    # demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    # Since we only want the statistics step to be run,
    # Pop the coregistration step of the cfg
    test_cfg.pop("coregistration")
    # Since we only want the input_ref, pop the input_sec
    # of the cfg
    test_cfg.pop("input_sec")
    test_cfg["statistics"]["alti-diff"]["classification_layers"].pop("Fusion0")
    test_cfg["statistics"]["ref-curvature"] = test_cfg["statistics"].pop(
        "alti-diff"
    )
    # Warning: "ref-curvature" does not work
    # with "Slope0" as "classification_layers"
    test_cfg["statistics"]["ref-curvature"]["classification_layers"].pop(
        "Slope0"
    )

    # Input configuration is
    # "input_ref": {
    #     "path": "./Gironde.tif",
    #     "zunit": "m",
    #     "classification_layers": {
    #        "Status": {
    #            "map_path": "./ref_status.tif"
    #        }
    # },
    # "input_sec": {
    #     "path": "./FinalWaveBathymetry_....TIF",
    #     "zunit": "m",
    #     "nodata": -32768,
    #     },
    # "statistics": {
    #     "remove_outliers": false,
    #     "classification_layers": {
    #         "Status": {
    #             "type": "segmentation",
    #            "classes": {"valid": [0],
    #                       "KO": [1],
    #                       "Land": [2],
    #                       "NoData": [3],
    #                       "Outside_detector": [4]}
    #         },
    #         "Slope0": {
    #              "type": "slope",
    #             "ranges": [0, 10, 25, 50, 90]
    #         }
    #     }
    # }

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        # Modify test's output dir in configuration to tmp test dir
        test_cfg["output_dir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "gironde_test_data"
        # configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)


@pytest.mark.end2end_tests
@pytest.mark.functional_tests
def test_demcompare_statistics_step_curvature_input_sec_with_gironde_test_data():  # noqa: E501, B950 # pylint: disable=line-too-long
    """
    Demcompare with only statistics step on one input dem
    end2end test.
    Input data:
    - Input dems and configuration present in the
      "gironde_test_data_sampling_sec/input" test data directory
    Validation process:
    - Reads the input configuration file
    - Deletes the input_sec of the configuration file
    - Deletes the coregistration step of the configuration file
    - Runs demcompare on a temporary directory
    - Checks that no error is raised
    """
    # Get "gironde_test_data" test root
    # data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data_sampling_ref")

    # Load "gironde_test_data_sampling_sec"
    # demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    # Since we only want the input_sec, pop the input_sec
    # of the cfg
    test_cfg["statistics"]["alti-diff"]["classification_layers"].pop("Fusion0")
    test_cfg["statistics"]["sec-curvature"] = test_cfg["statistics"].pop(
        "alti-diff"
    )
    # Warning: "sec-curvature" does not work with "Slope0" as "classification_layers" # noqa: E501, B950 # pylint: disable=line-too-long
    test_cfg["statistics"]["sec-curvature"]["classification_layers"].pop(
        "Slope0"
    )

    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        # Modify test's output dir in configuration to tmp test dir
        test_cfg["output_dir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, test_cfg)

        # Run demcompare with "gironde_test_data"
        # configuration (and replace conf file)
        demcompare.run(tmp_cfg_file)


@pytest.mark.end2end_tests
@pytest.mark.functional_tests
def test_initialization_with_wrong_classification_layers():
    """
    Test that demcompare's initialization raises an error
    when the input classification layer mask
    of the dem has a different size than its support dem
    Demcompare with only statistics step on one input dem
    end2end test.
    Input data:
    - Input dems and configuration present in the
      "gironde_test_data/input" test data directory
    Validation process:
    - Reads the input configuration file
    - Modifies the map_path of the input_sec classification layer
      to a different-sized mask
    - Runs demcompare on a temporary directory
    - Checks that an error is raised
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Set output_dir correctly
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)

        cfg["input_sec"]["classification_layers"]["Status"]["map_path"] = (
            os.path.join(
                test_data_path,
                "input",
                "Small_FinalWaveBathymetry_T30TXR_20200622T105631_Status.TIF",
            )
        )
        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, cfg)

        with pytest.raises(ValueError):
            # Compute initialization with wrong masks -> waits for a ValueError
            _ = compute_initialization(tmp_cfg_file)
