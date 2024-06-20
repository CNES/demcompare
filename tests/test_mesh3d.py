#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2023 CNES.
#
# This file is part of cars-mesh
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
"""Tests for cars_mesh package."""
# to check the code structure and avoid cyclic import !!

# Standard imports
import os
from tempfile import TemporaryDirectory

# Third party imports
import pytest

# cars_mesh imports
from cars_mesh import param, reconstruct_pipeline
from cars_mesh.config import read_config, save_config_file
from cars_mesh.tools.handlers import read_input_path

# Tests helpers
from .helpers import get_temporary_dir, get_test_data_path


# Filter rasterio warning when image is not georeferenced
@pytest.mark.filterwarnings(
    "ignore: Dataset has no geotransform, gcps, or rpcs"
)
@pytest.mark.end2end_tests
@pytest.mark.fast
def test_example_end2end():
    """
    Test one run on a fast end2end test.
    Run example configuration reconstruction (same from doc quickstart example).
    TODO : check the outputs !
    only run failed, no data verification.
    """
    # Get "toulouse_test_data" test root data
    # directory absolute path
    test_data_path = get_test_data_path("toulouse_test_data")

    # Load a reconstruct configuration example config
    test_cfg_path = os.path.join(
        test_data_path, "example_config_reconstruct.json"
    )

    # Create temporary directory for test output
    with TemporaryDirectory(dir=get_temporary_dir()) as tmp_dir:

        # Open config, Modify test's output dir to tmp test dir, save config
        test_cfg = read_config(test_cfg_path)
        test_cfg["output_dir"] = tmp_dir
        test_cfg_tmp_path = os.path.join(
            tmp_dir, os.path.basename(test_cfg_path)
        )
        save_config_file(test_cfg_tmp_path, test_cfg)

        # run cars-mesh main reconstruct pipeline in tmp dir
        reconstruct_pipeline.main(test_cfg_tmp_path)


# Filter rasterio warning when image is not georeferenced
@pytest.mark.filterwarnings(
    "ignore: Dataset has no geotransform, gcps, or rpcs"
)
@pytest.mark.functional_tests
@pytest.mark.slow
def test_all_possible_combinations():
    """
    Test all the possible combinations directly from methods
    except poisson reconstruction (cf documentation and readme)

    Doesn't test the state machine and the main CLI.
    Only each algorithm steps manually !

    """
    # Get "toulouse_test_data" test root data
    # directory absolute path
    test_data_path = get_test_data_path("toulouse_test_data")

    # Load a config_tests json config containing all sub steps configuration
    test_cfg_path = os.path.join(test_data_path, "config_tests.json")

    # open json config file to run automatically the tests
    test_cfg = read_config(test_cfg_path)

    # Create temporary directory for test output
    with TemporaryDirectory(dir=get_temporary_dir()) as tmp_dir:
        # Modify test's output dir in configuration to tmp test dir
        test_cfg["output_dir"] = tmp_dir

        # Loop on all steps listed in param.TRANSITION_METHODS:
        # filtering, denoising, meshing, simplify meshing, texturing

        # Loop on filtering methods
        for filtering_method in param.TRANSITIONS_METHODS["filter"]:
            # Loop on denoising methods
            for denoise_method in param.TRANSITIONS_METHODS["denoise_pcd"]:
                # Loop on meshing methods
                for mesh_method in param.TRANSITIONS_METHODS["mesh"]:
                    # Do not test poisson reconstruction because it changes the
                    # points' positions and thus creates outliers that make the
                    # texturing step fail
                    if mesh_method == "poisson":
                        continue

                    # Loop on mesh simplification methods
                    for simplify_mesh_method in param.TRANSITIONS_METHODS[
                        "simplify_mesh"
                    ]:
                        # Loop on texturing methods
                        for texture_method in param.TRANSITIONS_METHODS[
                            "texture"
                        ]:
                            # for each transition methods,
                            # run the pipeline steps.

                            # get point cloud in generic Mesh() structure
                            mesh_data = read_input_path(test_cfg["input_path"])

                            # Do filtering step
                            mesh_data.pcd = param.TRANSITIONS_METHODS["filter"][
                                filtering_method
                            ](
                                mesh_data.pcd,
                                **test_cfg[filtering_method]["params"]
                            )

                            # Do denoise step
                            mesh_data.pcd = param.TRANSITIONS_METHODS[
                                "denoise_pcd"
                            ][denoise_method](
                                mesh_data.pcd,
                                **test_cfg[denoise_method]["params"]
                            )

                            # Do mesh step (except poisson)
                            mesh_data = param.TRANSITIONS_METHODS["mesh"][
                                mesh_method
                            ](mesh_data.pcd, **test_cfg[mesh_method]["params"])

                            # Do simplify mesh step
                            mesh_data = param.TRANSITIONS_METHODS[
                                "simplify_mesh"
                            ][simplify_mesh_method](
                                mesh_data,
                                **test_cfg[simplify_mesh_method]["params"]
                            )

                            # do texturing step
                            mesh_data = param.TRANSITIONS_METHODS["texture"][
                                texture_method
                            ](mesh_data, test_cfg)
