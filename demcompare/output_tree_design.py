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
"""
Output tree design architecture for demcompare
"""

# Standard imports
import os

# In what comes next : OTD stands for Output Tree Design
default_OTD = {
    # first seen output
    "initial_dh.tif": ".",
    "final_dh.tif": ".",
    "final_config.json": ".",
    "dh_col_wise_wave_detection.tif": ".",
    "dh_row_wise_wave_detection.tif": ".",
    # coreg step
    "coreg_DEM.tif": "./coregistration/",
    "coreg_REF.tif": "./coregistration/",
    "nuth_kaab_tmp_dir": "./coregistration/nuth_kaab_tmp_dir",
    # snapshots
    "snapshots_dir": "./snapshots",
    "initial_dem_diff.png": "./snapshots/",
    "final_dem_diff.png": "./snapshots/",
    # histograms
    "histograms_dir": "./histograms",
    # stats
    "stats_dir": "./stats",
    "DSM_support.tif": "./stats",
    "Ref_support.tif": "./stats",
    "Ref_support-DSM_support.tif": "./stats",
    "DSM_support_classified.png": "./stats",
    "Ref_support_classified.png": "./stats",
    # doc
    "sphinx_built_doc": "./doc/published_report",
    "sphinx_src_doc": "./doc/src",
}

raw_OTD = {
    # first seen output
    "initial_dh.tif": ".",
    "final_dh.tif": ".",
    "final_config.json": ".",
    "dh_col_wise_wave_detection.tif": ".",
    "dh_row_wise_wave_detection.tif": ".",
    # coreg step
    "coreg_DEM.tif": ".",
    "coreg_REF.tif": ".",
    "nuth_kaab_tmp_dir": ".",
    # snapshots
    "snapshots_dir": ".",
    "initial_dem_diff.png": ".",
    "final_dem_diff.png": ".",
    # histograms
    "histograms_dir": ".",
    # stats
    "stats_dir": ".",
    "DSM_support.tif": ".",
    "Ref_support.tif": ".",
    "Ref_support-DSM_support.tif": "./stats",
    "DSM_support_classified.png": ".",
    "Ref_support_classified.png": ".",
    # doc
    "sphinx_built_doc": "./report_documentation",
    "sphinx_src_doc": "./tmpDir",
}


supported_OTD = {"raw_OTD": raw_OTD, "default_OTD": default_OTD}


def get_otd_dirs(design="default_OTD"):
    return list(set(supported_OTD[design].values()))


def get_out_dir(key, design="default_OTD"):
    return supported_OTD[design][key]


def get_out_file_path(key, design="default_OTD"):
    return os.path.join(get_out_dir(key, design), key)
