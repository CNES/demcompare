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
demcompare aims at coregistering and comparing two dems
"""


import copy
import json
import logging
import logging.config
import os
import shutil
import sys

import matplotlib as mpl
import numpy as np

from . import coregistration, initialization, report, stats
from .img_tools import load_dems, read_img, read_img_from_array, save_tif
from .output_tree_design import get_otd_dirs, get_out_dir, get_out_file_path

## VERSION
# Depending on python version get importlib standard lib or backported package
if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError  # pragma: no cover
    from importlib.metadata import version
else:
    from importlib_metadata import PackageNotFoundError  # pragma: no cover
    from importlib_metadata import version
try:
    dist_name = "demcompare"
    __version__ = version(dist_name)
except PackageNotFoundError:
    __version__ = "unknown"  # pragma: no cover
finally:
    del version, PackageNotFoundError

## STEPS
DEFAULT_STEPS = ['coregistration', 'stats', 'report']
ALL_STEPS = copy.deepcopy(DEFAULT_STEPS)

def setup_logging(path='demcompare/logging.json', default_level=logging.WARNING,):
    """
    Setup the logging configuration

    :param path: path to the configuration file
    :type path: string
    :param default_level: default level
    :type default_level: logging level
    """
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def computeReport(cfg, steps, dem, ref):
    """
    Create html and pdf report

    :param cfg: configuration dictionary
    :param dem: A3DDEMRaster, dem raster
    :param ref: A3DDEMRaster,reference dem raster to be coregistered to dem raster
    :return:
    """
    if 'report' in steps:
        report.generate_report(cfg['outputDir'],
                               dem.ds_file,
                               ref.ds_file,
                               cfg['stats_results']['partitions'],
                               os.path.join(cfg['outputDir'], get_out_dir('sphinx_built_doc')),
                               os.path.join(cfg['outputDir'], get_out_dir('sphinx_src_doc')))


def computeStats(cfg, dem, ref, final_dh, display=False, final_json_file=None):
    """
    Compute Stats on final_dh

    :param cfg: configuration dictionary
    :param dem: A3DDEMRaster, dem raster
    :param ref: A3DDEMRaster,reference dem raster to be coregistered to dem raster
    :param final_dh: xarray.Dataset, initial alti diff
    :param display: boolean, choose between plot show and plot save
    :param final_json_file: filename of final_cfg
    :return:
    """

    cfg['stats_results'] = {}
    cfg['stats_results']['images'] = {}
    cfg['stats_results']['images']['list'] = []

    stats.wave_detection(cfg, final_dh, display=display)

    stats.alti_diff_stats(cfg, dem, ref, final_dh, display=display, remove_outliers=cfg['stats_opts']['remove_outliers'])
    # save results
    with open(final_json_file, 'w') as outfile:
        json.dump(cfg, outfile, indent=2)


def computeCoregistration(cfg, steps, dem, ref, initial_dh, final_cfg=None, final_json_file=None):
    """
    Coregister two DSMs together and compute alti differences (before and after coregistration).

    This can be view as a two step process:
    - plani rectification computation
    - alti differences computation

    :param cfg: configuration dictionary
    :param dem: xarray.Dataset, dem raster
    :param ref: xarray.Dataset,reference dem raster to be coregistered to dem raster
    :param initial_dh: xarray.Dataset, inital alti diff
    :param final_cfg: cfg from a previous run
    :param final_json_file: filename of final_cfg
    :return: coregistered dem and ref A3DEMRAster + final alti differences as A3DGeoRaster
    """
    if 'coregistration' in steps:
        coreg_dem, coreg_ref, final_dh = coregistration.coregister_and_compute_alti_diff(cfg, dem, ref)

        # saves results here in case next step fails
        with open(final_json_file, 'w') as outfile:
            json.dump(cfg, outfile, indent=2)

    else:
        if final_cfg and 'plani_results' and 'alti_results' in final_cfg:
            cfg['plani_results'] = final_cfg['plani_results']
            cfg['alti_results'] = final_cfg['alti_results']
            coreg_dem = read_img(str(cfg['alti_results']['rectifiedDSM']['path']), no_data=(
                cfg['alti_results']['rectifiedDSM']['nodata'] if 'nodata' in cfg['alti_results'][
                    'rectifiedDSM'] else None))
            coreg_ref = read_img(str(cfg['alti_results']['rectifiedRef']['path']), no_data=(
                cfg['alti_results']['rectifiedRef']['nodata'] if 'nodata' in cfg['alti_results'][
                    'rectifiedRef'] else None))
            final_dh = read_img(str(cfg['alti_results']['dzMap']['path']),
                                    no_data=cfg['alti_results']['dzMap']['nodata'])

        else:
            coreg_ref = ref
            coreg_dem = dem
            final_dh = initial_dh
            cfg['plani_results'] = {}
            cfg['plani_results']['dx'] = {'bias_value': 0,
                                          'unit': 'm'}
            cfg['plani_results']['dy'] = {'bias_value': 0,
                                          'unit': 'm'}
            cfg['alti_results'] = {}
            cfg['alti_results']['rectifiedDSM'] = copy.deepcopy(cfg['inputDSM'])
            cfg['alti_results']['rectifiedRef'] = copy.deepcopy(cfg['inputRef'])

            coreg_dem = save_tif(coreg_dem, os.path.join(cfg['outputDir'], get_out_file_path('coreg_DEM.tif')))
            coreg_ref = save_tif(coreg_ref, os.path.join(cfg['outputDir'], get_out_file_path('coreg_REF.tif')))
            final_dh = save_tif(final_dh, os.path.join(cfg['outputDir'], get_out_file_path('final_dh.tif')))
            cfg['alti_results']['rectifiedDSM']['path'] = coreg_dem.attrs['ds_file']
            cfg['alti_results']['rectifiedRef']['path'] = coreg_ref.attrs['ds_file']
            cfg['alti_results']['rectifiedDSM']['nb_points'] = coreg_dem['im'].data.size
            cfg['alti_results']['rectifiedRef']['nb_points'] = coreg_ref['im'].data.size
            cfg['alti_results']['rectifiedDSM']['nb_valid_points'] = np.count_nonzero(~np.isnan(coreg_dem['im'].data))
            cfg['alti_results']['rectifiedRef']['nb_valid_points'] = np.count_nonzero(~np.isnan(coreg_ref['im'].data))
            cfg['alti_results']['dzMap'] = {'path': final_dh.attrs['ds_file'],
                                            'zunit': coreg_ref.attrs['zunit'].name,
                                            'nodata': final_dh.coords["no_data"],
                                            'nb_points': final_dh['im'].data.size,
                                            'nb_valid_points': np.count_nonzero(~np.isnan(final_dh['im'].data.size))}

    return coreg_dem, coreg_ref, final_dh


def computeInitialization(config_json):

    # read the json configuration file
    with open(config_json, 'r') as f:
        cfg = json.load(f)

    # create output directory
    cfg['outputDir'] = os.path.abspath(cfg['outputDir'])
    initialization.mkdir_p(cfg['outputDir'])

    # copy config into outputDir
    try:
        shutil.copy(config_json,
                    os.path.join(cfg['outputDir'], os.path.basename(config_json)))
    except shutil.Error:
        # file exists or file is the same
        pass
    except:
        raise

    # checks config
    initialization.check_parameters(cfg)

    #create output tree dirs
    [initialization.mkdir_p(os.path.join(cfg['outputDir'], directory)) for directory in get_otd_dirs(cfg['otd'])]

    initialization.initialization_plani_opts(cfg)
    initialization.initialization_alti_opts(cfg)
    initialization.initialization_stats_opts(cfg)

    return cfg


def run_tile(json_file, steps=DEFAULT_STEPS, display=False, debug=False, force=False):
    """
    demcompare execution for a single tile

    :param json_file:
    :param steps:
    :param display:
    :param debug:
    :param force:
    :return:
    """

    #
    # Initialization
    #
    cfg = computeInitialization(json_file)
    print(('*** demcompare : start processing into {} ***'.format(cfg['outputDir'])))
    sys.stdout.flush()
    if display is False:
        # if display is False we have to tell matplotlib to cancel it
        mpl.use('Agg')

    # Set final_json_file name and try to read it if it exists (if a previous run was launched)
    final_json_file = os.path.join(cfg['outputDir'], get_out_file_path('final_config.json'))
    final_cfg = None
    if os.path.isfile(final_json_file):
        with open(final_json_file, 'r') as f:
            final_cfg = json.load(f)

    #
    # Create datasets
    #
    ref, dem = load_dems(cfg['inputRef']['path'], cfg['inputDSM']['path'],
                             ref_nodata=(cfg['inputRef']['nodata'] if 'nodata' in cfg['inputRef'] else None),
                             dem_nodata=(cfg['inputDSM']['nodata'] if 'nodata' in cfg['inputDSM'] else None),
                             ref_georef=cfg['inputRef']['georef'], dem_georef=cfg['inputDSM']['georef'],
                             ref_zunit=(cfg['inputRef']['zunit'] if 'zunit' in cfg['inputRef'] else 'm'),
                             dem_zunit=(cfg['inputDSM']['zunit'] if 'zunit' in cfg['inputDSM'] else 'm'),
                             load_data=(cfg['roi'] if 'roi' in cfg else True))

    #
    # Compute initial dh and save it
    #

    initial_dh = read_img_from_array(ref['im'].data - dem['im'].data, from_dataset=dem, no_data=-32768)
    initial_dh = save_tif(initial_dh, os.path.join(cfg['outputDir'], get_out_file_path('initial_dh.tif')))

    stats.dem_diff_plot(initial_dh, title='DSMs diff without coregistration (REF - DSM)',
                        plot_file=os.path.join(cfg['outputDir'], get_out_file_path('initial_dem_diff.png')),
                        display=False)

    #
    # Coregister both DSMs together and compute final differences
    #
    coreg_dem, coreg_ref, final_dh = computeCoregistration(cfg, steps, dem, ref, initial_dh,
                                                           final_cfg=final_cfg, final_json_file=final_json_file)
    if final_dh is not initial_dh:
        stats.dem_diff_plot(final_dh, title='DSMs diff with coregistration (REF - DSM)',
                            plot_file=os.path.join(cfg['outputDir'], get_out_file_path('final_dem_diff.png')),
                            display=False)

    #
    # Compute stats
    #
    computeStats(cfg, coreg_dem, coreg_ref, final_dh, display=display, final_json_file=final_json_file)

    #
    # Compute reports
    #
    computeReport(cfg, steps, coreg_dem, coreg_ref)


def run(json_file, steps=DEFAULT_STEPS, display=False, debug=False, force=False):
    #
    # Initialization
    #
    setup_logging()
    cfg = computeInitialization(json_file)
    if display is False:
        # if display is False we have to tell matplotlib to cancel it
        mpl.use('Agg')

    #
    # Get back tiles
    #
    if 'tile_size' not in cfg:
        tiles = [{'json': json_file}]
    else:
        tiles = initialization.divide_images(cfg)

    #
    # Run classic steps by tiles (there can be just one tile which could be the whole image)
    #
    for tile in tiles:
        try:
            run_tile(tile['json'], steps, display=display, debug=debug, force=force)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(('Error encountered for tile: {} -> {}'.format(tile, e)))
            pass
