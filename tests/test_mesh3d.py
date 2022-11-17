#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 CNES.
#
# This file is part of mesh3d
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
"""Tests for `mesh3d` package."""

import json

# other import
import os
import shutil

# Third party imports
import pytest

# mesh3d imports
import mesh3d
from mesh3d import param
from mesh3d.tools.handlers import read_input_path


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # Example to edit
    return "response"


def test_content(response):  # pylint: disable=redefined-outer-name
    """Sample pytest test function with the pytest fixture as an argument."""
    # Example to edit
    print(response)


def test_mesh3d():
    """Sample pytest mesh3d module test function"""
    assert mesh3d.__author__ == "CNES"
    assert mesh3d.__email__ == "cars@cnes.fr"


# Poisson recontruction tests are missing (cf documentation and readme)
def test_all_possible_combinations():
    """Test all the possible combinations (except poisson reconstruction)"""
    # open config file to launch automatically the tests
    with open("tests/config_tests.json", "r", encoding="utf-8") as cfg_file:
        cfg = json.load(cfg_file)

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
                    for texture_method in param.TRANSITIONS_METHODS["texture"]:

                        mesh_data = read_input_path(cfg["input_path"])
                        mesh_data.pcd = param.TRANSITIONS_METHODS["filter"][
                            filtering_method
                        ](mesh_data.pcd, **cfg[filtering_method]["params"])
                        mesh_data.pcd = param.TRANSITIONS_METHODS[
                            "denoise_pcd"
                        ][denoise_method](
                            mesh_data.pcd, **cfg[denoise_method]["params"]
                        )
                        mesh_data = param.TRANSITIONS_METHODS["mesh"][
                            mesh_method
                        ](mesh_data.pcd, **cfg[mesh_method]["params"])
                        mesh_data = param.TRANSITIONS_METHODS["simplify_mesh"][
                            simplify_mesh_method
                        ](mesh_data, **cfg[simplify_mesh_method]["params"])
                        os.makedirs("example/output", exist_ok=True)
                        mesh_data = param.TRANSITIONS_METHODS["texture"][
                            texture_method
                        ](mesh_data, cfg)
                        shutil.rmtree("example/output")
