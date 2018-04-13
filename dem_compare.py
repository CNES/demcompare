#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

"""
dem_compare aims at coregistering and comparing two dsms


"""

from __future__ import print_function
import os
import sys
import errno
import json
import argparse
import shutil
from osgeo import gdal
import numpy as np
import copy
import csv
import matplotlib as mpl
from a3d_modules.a3d_georaster import A3DDEMRaster, A3DGeoRaster
from dem_compare_lib import initialization, coregistration, stats, report


gdal.UseExceptions()
DEFAULT_STEPS = ['coregistration', 'stats', 'report']
ALL_STEPS = copy.deepcopy(DEFAULT_STEPS)

###############
#configuration#
def mkdir_p(path):
    """
    Create a directory without complaining if it already exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc: # requires Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_tile_dir(cfg, c, r, w, h):
    """
    Get the name of a tile directory
    """
    max_digit_row = 0
    max_digit_col = 0
    if 'max_digit_tile_row' in cfg:
        max_digit_row = cfg['max_digit_tile_row']
    if 'max_digit_tile_col' in cfg:
        max_digit_col = cfg['max_digit_tile_col']
    return os.path.join(cfg['outputDir'],
                        'tiles',
                        'row_{:0{}}_height_{}'.format(r, max_digit_row, h),
                        'col_{:0{}}_width_{}'.format(c, max_digit_col, w))


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
                               os.path.join(cfg['outputDir'], 'report_documentation'),
                               os.path.join(cfg['outputDir'], 'tmpDir'))


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
            coreg_dem.save_geotiff(os.path.join(cfg['outputDir'], 'coreg_DEM.tif'))
            coreg_ref.save_geotiff(os.path.join(cfg['outputDir'], 'coreg_REF.tif'))
            final_dh.save_geotiff(os.path.join(cfg['outputDir'], 'final_dh.tif'))
            cfg['alti_results']['rectifiedDSM']['path'] = coreg_dem.ds_file
            cfg['alti_results']['rectifiedRef']['path'] = coreg_ref.ds_file
            cfg['alti_results']['rectifiedDSM']['nb_points'] = coreg_dem.r.size
            cfg['alti_results']['rectifiedRef']['nb_points'] = coreg_ref.r.size
            cfg['alti_results']['rectifiedDSM']['nb_valid_points'] = np.count_nonzero(~np.isnan(coreg_dem.r))
            cfg['alti_results']['rectifiedRef']['nb_valid_points'] = np.count_nonzero(~np.isnan(coreg_ref.r))

    return coreg_dem, coreg_ref, final_dh


def computeInitialization(config_json):

    # read the json configuration file
    with open(config_json, 'r') as f:
        cfg = json.load(f)

    # create output directory
    cfg['outputDir'] = os.path.abspath(cfg['outputDir'])
    mkdir_p(cfg['outputDir'])

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

    # set tmpDir (trash dir)
    cfg['tmpDir'] = os.path.join(cfg['outputDir'], 'tmp')
    mkdir_p(cfg['tmpDir'])
    # if "clean_tmp" not in cfg or cfg["clean_tmp"] is True:
        # if "TMPDIR" not in os.environ:
        #     os.environ["TMPDIR"] = os.path.join(cfg['outputDir'], "tmp")
        #     try:
        #         os.makedirs(os.environ["TMPDIR"])
        #     except OSError as e:
        #         if e.errno != errno.EEXIST:
        #             raise
        # else :
        #     cfg['tmpDir'] = os.path.expandvars("$TMPDIR")

    initialization.initialization_plani_opts(cfg)
    initialization.initialization_alti_opts(cfg)
    initialization.initialization_stats_opts(cfg)

    return cfg


def main(json_file, steps=DEFAULT_STEPS, display=False, debug=False, force=False):
    #
    # Initialization
    #
    cfg = computeInitialization(json_file)
    print('*** dem_compare.PY : start processing into {} ***'.format(cfg['outputDir']))
    sys.stdout.flush()
    if display is False:
        # if display is False we have to tell matplotlib to cancel it
        mpl.use('Agg')

    # Only now import a3d_georaster classes since they rely on matplotlib
    from a3d_modules.a3d_georaster import A3DDEMRaster, A3DGeoRaster

    # Set final_json_file name and try to read it if it exists (if a previous run was launched)
    final_json_file = os.path.join(cfg['outputDir'], 'final_config.json')
    final_cfg = None
    if os.path.isfile(final_json_file):
        with open(final_json_file, 'r') as f:
            final_cfg = json.load(f)

    #
    # Create A3DDEMRaster
    #
    dem = A3DDEMRaster(cfg['inputDSM']['path'], nodata=(cfg['inputDSM']['nodata'] if 'nodata' in cfg['inputDSM'] else None),
                       load_data=(cfg['roi'] if 'roi' in cfg else True), ref=cfg['inputDSM']['georef'],
                       zunit=(cfg['inputDSM']['zunit'] if 'zunit' in cfg['inputDSM'] else 'm'))
    ref = A3DDEMRaster(cfg['inputRef']['path'], nodata=(cfg['inputRef']['nodata'] if 'nodata' in cfg['inputRef'] else None),
                       load_data=(cfg['roi'] if 'roi' in cfg else True), ref=cfg['inputRef']['georef'],
                       zunit=(cfg['inputRef']['zunit'] if 'zunit' in cfg['inputRef'] else 'm'))
    nodata1 = dem.nodata
    nodata2 = ref.nodata

    # test if roi is invalid
    if (np.count_nonzero(np.isnan(ref.r)) == ref.r.size) or (np.count_nonzero(np.isnan(dem.r)) == dem.r.size):
        raise Exception("The ROI contents all nodata pixels. Change x/y of the ROI.")

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

    #
    # Compute initial dh and save it
    #
    initial_dh = A3DGeoRaster.from_raster(reproj_dem.r - reproj_ref.r,
                                          reproj_dem.trans,
                                          "{}".format(reproj_dem.srs.ExportToProj4()),
                                          nodata=-32768)
    initial_dh.save_geotiff(os.path.join(cfg['outputDir'],'initial_dh.tif'))
    stats.dem_diff_plot(initial_dh, title='DSMs diff without coregistration',
                        plot_file=os.path.join(cfg['outputDir'],'initial_dem_diff.png'), display=False)

    #
    # Coregister both DSMs together and compute final differences
    #
    coreg_dem, coreg_ref, final_dh = computeCoregistration(cfg, steps, reproj_dem, reproj_ref, initial_dh,
                                                           final_cfg= final_cfg, final_json_file=final_json_file)
    if final_dh is not initial_dh:
        stats.dem_diff_plot(final_dh, title='DSMs diff with coregistration',
                            plot_file=os.path.join(cfg['outputDir'],'final_dem_diff.png'), display=False)

    #
    # Compute stats
    #
    computeStats(cfg, coreg_dem, coreg_ref, final_dh, display=display, final_json_file=final_json_file)

    #
    # Compute reports
    #
    computeReport(cfg, steps, coreg_dem, coreg_ref)


def get_parser():
    """
    ArgumentParser for dem_compare
    :param None
    :return parser
    """
    parser = argparse.ArgumentParser(description=('Compares DSMs'))

    parser.add_argument('config', metavar='config.json',
                        help=('path to a json file containing the paths to '
                              'input and output files and the algorithm '
                              'parameters'))
    parser.add_argument('--step', type=str, nargs='+', choices=ALL_STEPS,
                        default=DEFAULT_STEPS)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--display', action='store_true')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.config, args.step, debug=args.debug, display=args.display)

