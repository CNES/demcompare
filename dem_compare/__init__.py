#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Copyright (C) 2017-2018 Centre National d'Etudes Spatiales (CNES)

"""
dem_compare aims at coregistering and comparing two dsms


"""

from __future__ import print_function
import os
import sys
import json
import argparse
import shutil
from osgeo import gdal
import numpy as np
import copy
import matplotlib as mpl
import initialization, coregistration, stats, report, dem_compare_extra
from output_tree_design import get_out_dir, get_out_file_path, get_otd_dirs
from a3d_georaster import A3DDEMRaster, A3DGeoRaster


gdal.UseExceptions()
DEFAULT_STEPS = ['coregistration', 'stats', 'report'] + dem_compare_extra.DEFAULT_STEPS
ALL_STEPS = copy.deepcopy(DEFAULT_STEPS) + dem_compare_extra.ALL_STEPS


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
                               [modename for modename, modefile in cfg['stats_results']['modes'].items()],
                               os.path.join(cfg['outputDir'], get_out_dir('sphinx_built_doc')),
                               os.path.join(cfg['outputDir'], get_out_dir('sphinx_src_doc')))


def computeStats(cfg, dem, ref, final_dh, display=False, final_json_file=None):
    """
    Compute Stats on final_dh

    :param cfg: configuration dictionary
    :param dem: A3DDEMRaster, dem raster
    :param ref: A3DDEMRaster,reference dem raster to be coregistered to dem raster
    :param final_dh: A3DGeoRaster, inital alti diff
    :param display: boolean, choose between plot show and plot save
    :param final_json_file: filename of final_cfg
    :return:
    """

    cfg['stats_results'] = {}
    cfg['stats_results']['images'] = {}
    cfg['stats_results']['images']['list'] = []

    stats.wave_detection(cfg, final_dh, display=display)

    stats.alti_diff_stats(cfg, dem, ref, final_dh, display=display)
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
    :param dem: A3DDEMRaster, dem raster
    :param ref: A3DDEMRaster,reference dem raster to be coregistered to dem raster
    :param initial_dh: A3DGeoRaster, inital alti diff
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
            coreg_dem = A3DDEMRaster(str(cfg['alti_results']['rectifiedDSM']['path']), nodata=(cfg['alti_results']['rectifiedDSM']['nodata'] if 'nodata' in cfg['alti_results']['rectifiedDSM'] else None))
            coreg_ref = A3DDEMRaster(str(cfg['alti_results']['rectifiedRef']['path']), nodata=(cfg['alti_results']['rectifiedRef']['nodata'] if 'nodata' in cfg['alti_results']['rectifiedRef'] else None))
            final_dh = A3DGeoRaster(str(cfg['alti_results']['dzMap']['path']), nodata=cfg['alti_results']['dzMap']['nodata'])
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
            coreg_dem.save_geotiff(os.path.join(cfg['outputDir'], get_out_file_path('coreg_DEM.tif')))
            coreg_ref.save_geotiff(os.path.join(cfg['outputDir'], get_out_file_path('coreg_REF.tif')))
            final_dh.save_geotiff(os.path.join(cfg['outputDir'], get_out_file_path('final_dh.tif')))
            cfg['alti_results']['rectifiedDSM']['path'] = coreg_dem.ds_file
            cfg['alti_results']['rectifiedRef']['path'] = coreg_ref.ds_file
            cfg['alti_results']['rectifiedDSM']['nb_points'] = coreg_dem.r.size
            cfg['alti_results']['rectifiedRef']['nb_points'] = coreg_ref.r.size
            cfg['alti_results']['rectifiedDSM']['nb_valid_points'] = np.count_nonzero(~np.isnan(coreg_dem.r))
            cfg['alti_results']['rectifiedRef']['nb_valid_points'] = np.count_nonzero(~np.isnan(coreg_ref.r))

    return coreg_dem, coreg_ref, final_dh


def load_dems(ref_path, dem_path, ref_nodata=None, dem_nodata=None,
              ref_georef='WGS84', dem_georef='WGS84', ref_zunit='m', dem_zunit='m', load_data=True):
    """
    Loads both DEMs

    :param ref_path: str, path to ref dem
    :param dem_path: str, path to sec dem
    :param ref_nodata: float, ref no data value (None by default and if set inside metadata)
    :param dem_nodata: float, dem no data value (None by default and if set inside metadata)
    :param ref_georef: str, ref georef (either WGS84 -default- or EGM96)
    :param dem_georef: str, dem georef (either WGS84 -default- or EGM96)
    :param ref_zunit: unit, ref z unit
    :param dem_zunit: unit, dem z unit
    :param load_data: True if dem are to be fully loaded, other options are False or a dict roi
    :return: ref and dem A3DDEMRaster
    """

    #
    # Create A3DDEMRaster
    #
    dem = A3DDEMRaster(dem_path, nodata=dem_nodata, load_data=load_data, ref=dem_georef, zunit=dem_zunit)
    # the ref dem is read according to the tested dem roi
    # -> this means first we get back ref footprint by only reading metadata (load_data is False)
    ref = A3DDEMRaster(ref_path,
                       load_data=False)
    # -> then we compute the common footprint between dem roi and ref
    ref_matching_footprint = ref.biggest_common_footprint(dem)
    # -> finally we add a marge to this footprint because data will be loaded from image indexes
    #    and there is no bijection from pixels indexes to footprint
    ref_matching_roi = ref.footprint_to_roi(ref_matching_footprint)
    ref_matching_roi['x'] -= 1
    ref_matching_roi['y'] -= 1
    ref_matching_roi['w'] += 1
    ref_matching_roi['h'] += 1
    ref = A3DDEMRaster(ref_path, nodata=ref_nodata, load_data=ref_matching_roi, ref=ref_georef, zunit=ref_zunit)
    # test if roi is invalid
    if (np.count_nonzero(np.isnan(ref.r)) == ref.r.size) or (np.count_nonzero(np.isnan(dem.r)) == dem.r.size):
        raise Exception("The ROI covers nodata pixels only. Please change ROI description then try again.")

    nodata1 = dem.nodata
    nodata2 = ref.nodata

    #
    # Reproject DSMs
    #
    biggest_common_grid = dem.biggest_common_footprint(ref)
    nx = (biggest_common_grid[1] - biggest_common_grid[0]) / dem.xres
    ny = (biggest_common_grid[2] - biggest_common_grid[3]) / dem.yres
    reproj_dem = dem.reproject(dem.srs, int(nx), int(ny), biggest_common_grid[0], biggest_common_grid[3],
                               dem.xres, dem.yres, nodata=nodata1)
    reproj_ref = ref.reproject(dem.srs, int(nx), int(ny), biggest_common_grid[0], biggest_common_grid[3],
                               dem.xres, dem.yres, nodata=nodata2)
    reproj_dem.r[reproj_dem.r == nodata1] = np.nan
    reproj_ref.r[reproj_ref.r == nodata2] = np.nan

    return reproj_ref, reproj_dem


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
    dem_compare execution for a single tile

    :param json_file:
    :param steps:
    :param display:
    :param debug:
    :param force:
    :return:
    """
    if all(step in dem_compare_extra.ALL_STEPS for step in steps):
        return

    #
    # Initialization
    #
    cfg = computeInitialization(json_file)
    print('*** dem_compare.py : start processing into {} ***'.format(cfg['outputDir']))
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
    # Create A3DDEMRaster
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
    initial_dh = A3DGeoRaster.from_raster(ref.r - dem.r,
                                          dem.trans,
                                          "{}".format(dem.srs.ExportToProj4()),
                                          nodata=-32768)
    initial_dh.save_geotiff(os.path.join(cfg['outputDir'], get_out_file_path('initial_dh.tif')))
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
        except Exception, e:
            print('Error encountered for tile: {} -> {}'.format(tile, e))
            raise
            #TODO pass

    #
    # Run merge steps
    #
    if len(tiles) > 1:
        cfg['tiles_list_file'] = os.path.join(cfg['outputDir'], 'tiles.txt')
        dem_compare_extra.merge_tiles(cfg, steps, debug=debug, force=force)
