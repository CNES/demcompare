#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
methods in the demcompare_tiles module.
"""

import pytest

from demcompare.demcompare_tiles import verify_config


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    ["dict_config", "expected_error"],
    [
        pytest.param(
            {"height": "100", "width": 100, "overlap": 100, "nb_cpu": 1},
            "Height is not consistent",
        ),
        pytest.param(
            {"height": 100, "width": -100, "overlap": 100, "nb_cpu": 1},
            "Width is not consistent",
        ),
        pytest.param(
            {"height": 100, "width": 100, "nb_cpu": 1},
            "Overlap is not consistent",
        ),
        pytest.param(
            {"height": 100, "width": 100, "overlap": 100, "nb_cpu": -5},
            "Number of CPUs is incorrect",
        ),
        pytest.param(
            {"height": 100, "width": 100, "overlap": 100, "nb_cpu": 999999},
            "Number of CPUs in the config is more than available CPUs",
        ),
    ],
)
def test_verify_config(dict_config, expected_error):
    """
    Test the verify_config function
    Input data:
    - handcraft tiling parameter dictionary
    Validation data:
    - handcraft error message
    Validation process:
    - Check that the config dictionary does contain error
    - Checked function : verify_config
    """

    with pytest.raises(ValueError) as exc_info:
        verify_config(dict_config)

    assert str(exc_info.value) == expected_error
