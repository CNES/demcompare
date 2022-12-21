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
This module contains functions to test the Demcompare notebooks.
"""
import subprocess
import tempfile

import pytest

from .helpers import notebooks_demcompare_path


@pytest.mark.unit_tests
@pytest.mark.notebook_tests
def test_reprojection_and_coregistration():
    """
    Test that the reprojection_and_coregistration
    notebook runs without errors

    """
    reprojection_and_coregistration_path = notebooks_demcompare_path(
        "reprojection_and_coregistration.ipynb"
    )

    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                f"jupyter nbconvert --to script \
                {reprojection_and_coregistration_path} \
                --output-dir {directory}"
            ],
            shell=True,
            check=False,
        )
        out = subprocess.run(
            [f"ipython {directory}/reprojection_and_coregistration.py"],
            shell=True,
            check=False,
            cwd="../notebooks",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    assert out.returncode == 0


@pytest.mark.unit_tests
@pytest.mark.notebook_tests
def test_statistics():
    """
    Test that the reprojection_and_coregistration
    notebook runs without errors

    """
    statistics_path = notebooks_demcompare_path("statistics.ipynb")

    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                f"jupyter nbconvert --to script \
                    {statistics_path} --output-dir {directory}"
            ],
            shell=True,
            check=False,
        )
        out = subprocess.run(
            [f"ipython {directory}/statistics.py"],
            shell=True,
            check=False,
            cwd="../notebooks",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    assert out.returncode == 0


@pytest.mark.unit_tests
@pytest.mark.notebook_tests
def test_introduction_and_basic_usage():
    """
    Test that the introduction_and_basic_usage
    notebook runs without errors
    """
    introduction_and_basic_usage_path = notebooks_demcompare_path(
        "introduction_and_basic_usage.ipynb"
    )

    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                f"jupyter nbconvert --to script \
                {introduction_and_basic_usage_path} \
                --output-dir {directory}"
            ],
            shell=True,
            check=False,
        )
        out = subprocess.run(
            [f"ipython {directory}/introduction_and_basic_usage.py"],
            shell=True,
            check=False,
            cwd="../notebooks",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    assert out.returncode == 0
