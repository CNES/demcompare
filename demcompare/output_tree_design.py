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
"""
Output tree design architecture for demcompare
"""

# Standard imports
import os

# In what comes next : OTD stands for Output Tree Design
default_OTD = {
    # first seen output
    "initial_dem_diff.tif": ".",
    "final_dem_diff.tif": ".",
    "demcompare_results.json": ".",
    "dh_col_wise_wave_detection.tif": "./stats",
    "dh_row_wise_wave_detection.tif": "./stats",
    # coreg step
    "coreg_SEC.tif": "./coregistration/",
    "reproj_REF.tif": "./coregistration/",
    "reproj_SEC.tif": "./coregistration/",
    "reproj_coreg_REF.tif": "./coregistration/",
    "reproj_coreg_SEC.tif": "./coregistration/",
    "nuth_kaab_tmp_dir": "./coregistration/nuth_kaab_tmp_dir",
    # snapshots
    "snapshots_dir": "./snapshots",
    "initial_dem_diff.png": "./snapshots/",
    "initial_dem_diff_pdf.png": "./snapshots/",
    "initial_dem_diff_pdf.csv": "./snapshots/",
    "initial_dem_diff_cdf.png": "./snapshots/",
    "initial_dem_diff_cdf.csv": "./snapshots/",
    "final_dem_diff.png": "./snapshots/",
    "final_dem_diff_pdf.png": "./snapshots/",
    "final_dem_diff_pdf.csv": "./snapshots/",
    "final_dem_diff_cdf.png": "./snapshots/",
    "final_dem_diff_cdf.csv": "./snapshots/",
    # stats
    "_stats_dir": "./stats",
    "sec_support.tif": "./stats",
    "ref_support.tif": "./stats",
    "ref_support-sec_support.tif": "./stats",
    "DSM_support_classified.png": "./stats",
    "ref_support_classified.png": "./stats",
    "dem_for_stats.tif": "./stats",
    # doc
    "sphinx_built_doc": "./doc/published_report",
    "sphinx_src_doc": "./doc/src",
    "logs.log": ".",
}

supported_OTD = {"default_OTD": default_OTD}


def get_otd_dirs(design="default_OTD"):
    """Get All Output Tree Design directories"""
    return list(set(supported_OTD[design].values()))


def get_out_dir(key, design="default_OTD"):
    """Get key chosen Output Tree Design directory"""
    return supported_OTD[design][key]


def get_out_file_path(key, design="default_OTD"):
    """Get full path of get_out_dir from key in OTD"""
    return os.path.join(get_out_dir(key, design), key)
