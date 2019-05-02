#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Copyright (C) 2017-2018 Centre National d'Etudes Spatiales (CNES)

import os

# In what comes next : OTD stands for Output Tree Design
default_OTD = {
    #first seen output
    'initial_dh.tif': '.',
    'final_dh.tif': '.',
    'final_config.json': '.',
    'dh_col_wise_wave_detection.tif': '.',
    'dh_row_wise_wave_detection.tif': '.',
    #coreg step
    'coreg_DEM.tif': './coregistration/',
    'coreg_REF.tif': './coregistration/',
    'nuth_kaab_tmpDir': './coregistration/nuth_kaab_tmpDir',
    #snapshots
    'snapshots_dir': './snapshots',
    'initial_dem_diff.png': './snapshots/',
    'final_dem_diff.png': './snapshots/',
    #histograms
    'histograms_dir': './histograms',
    #stats
    'stats_dir': './stats',
    'DSM_support.tif': './stats',
    'Ref_support.tif': './stats',
    'Ref_support-DSM_support.tif': './stats',
    'DSM_support_classified.png': './stats',
    'Ref_support_classified.png': './stats',
    #doc
    'sphinx_built_doc': './doc/published_report',
    'sphinx_src_doc': './doc/src'
}

raw_OTD = {
    #first seen output
    'initial_dh.tif': '.',
    'final_dh.tif': '.',
    'final_config.json': '.',
    'dh_col_wise_wave_detection.tif': '.',
    'dh_row_wise_wave_detection.tif': '.',
    #coreg step
    'coreg_DEM.tif': '.',
    'coreg_REF.tif': '.',
    'nuth_kaab_tmpDir': '.',
    #snapshots
    'snapshots_dir': '.',
    'initial_dem_diff.png': '.',
    'final_dem_diff.png': '.',
    #histograms
    'histograms_dir': '.',
    #stats
    'stats_dir': '.',
    'DSM_support.tif': '.',
    'Ref_support.tif': '.',
    'Ref_support-DSM_support.tif': './stats',
    'DSM_support_classified.png': '.',
    'Ref_support_classified.png': '.',
    #doc
    'sphinx_built_doc': './report_documentation',
    'sphinx_src_doc': './tmpDir'
}


supported_OTD = {'raw_OTD': raw_OTD, 'default_OTD': default_OTD}


def get_otd_dirs(design='default_OTD'):
    return list(set(supported_OTD[design].values()))


def get_out_dir(key, design='default_OTD'):
    return supported_OTD[design][key]


def get_out_file_path(key, design='default_OTD'):
    return os.path.join(get_out_dir(key, design), key)