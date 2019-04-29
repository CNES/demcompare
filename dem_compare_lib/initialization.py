#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Copyright (C) 2017-2018 Centre National d'Etudes Spatiales (CNES)

"""
Init part of dsm_compare
This is where high level parameters are checked and default options are set

"""

from osgeo import gdal
from astropy import units as u
import ast
import numpy as np
import json
import copy
import errno
import os
from dem_compare_lib.a3d_georaster import A3DDEMRaster



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


def check_parameters(cfg):
    """
    Checks parameters
    """
    # verify that input files are defined
    if 'inputDSM' not in cfg or 'inputRef' not in cfg:
        raise NameError('ERROR: missing input images description')

    # verify if input files are correctly defined :
    if 'path' not in cfg['inputDSM'] or 'path' not in cfg['inputRef']:
        raise NameError('ERROR: missing paths to input images')
    cfg['inputDSM']['path'] = os.path.abspath(str(cfg['inputDSM']['path']))
    cfg['inputRef']['path'] = os.path.abspath(str(cfg['inputRef']['path']))


    # verify z units
    if 'zunit' not in cfg['inputDSM']:
        cfg['inputDSM']['zunit'] = 'meter'
    else:
        try:
            unit = u.Unit(cfg['inputDSM']['zunit'])
        except ValueError:
            raise NameError('ERROR: input DSM zunit ({}) not a supported unit'.format(cfg['inputDSM']['zunit']))
        if unit.physical_type != u.m.physical_type:
            raise NameError('ERROR: input DSM zunit ({}) not a lenght unit'.format(cfg['inputDSM']['zunit']))
    if 'zunit' not in cfg['inputRef']:
        cfg['inputRef']['zunit'] = 'meter'
    else:
        try:
            unit = u.Unit(cfg['inputRef']['zunit'])
        except ValueError:
            raise NameError('ERROR: input Ref zunit ({}) not a supported unit'.format(cfg['inputRef']['zunit']))
        if unit.physical_type != u.m.physical_type:
            raise NameError('ERROR: input Ref zunit ({}) not a lenght unit'.format(cfg['inputRef']['zunit']))

    # check ref:
    if 'geoid' in cfg['inputDSM'] or 'geoid' in cfg['inputRef']:
        print("WARNING : geoid option is deprecated. Use georef keyword now with EGM96 or WGS84 value")
    # what we do bellow is just in case someone used georef as geoid was used...
    if 'georef' in cfg['inputDSM']:
        if cfg['inputDSM']['georef'] is True:
            cfg['inputDSM']['georef'] = 'EGM96'
        else:
            if cfg['inputDSM']['georef'] is False:
                cfg['inputDSM']['georef'] = 'WGS84'
    else:
        cfg['inputDSM']['georef'] = 'WGS84'
    if 'georef' in cfg['inputRef']:
        if cfg['inputRef']['georef'] is True:
            cfg['inputRef']['georef'] = 'EGM96'
        else:
            if cfg['inputRef']['georef'] is False:
                cfg['inputRef']['georef'] = 'WGS84'
    else:
        cfg['inputRef']['georef'] = 'WGS84'


def initialization_plani_opts(cfg):
    """
    Initialize the plan2DShift step used to compute plani (x,y) shift between the two DSMs.
    'auto_disp_first_guess' : when set to True PRO_DecMoy is used to guess disp init and disp range
    'coregistration_method' : 'correlation' or 'nuth_kaab'
    if 'correlation' :
        'correlator' : 'PRO_Medicis'
        'disp_init' and 'disp_range' define the area to explore when 'auto_disp_first_guess' is set to False

    Note that disp_init and disp_range are used to define margin when the process is tiled.

    :param cfg:
    :return:
    """

    default_plani_opts = {'coregistration_method' : 'nuth_kaab',
                          'disp_init': {'x': 0, 'y': 0}}

    if 'plani_opts' not in cfg:
        cfg['plani_opts'] = default_plani_opts
    else:
        # we keep users items and add default items he has not set
        cfg['plani_opts'] = dict(default_plani_opts.items() + cfg['plani_opts'].items())


def initialization_alti_opts(cfg):
    default_alti_opts = {'egm96-15': {'path': "/work/logiciels/atelier3D/Data/egm/egm96_15.gtx", 'zunit': "meter"},
                         'deramping': False}

    if 'alti_opts' not in cfg:
        cfg['alti_opts'] = default_alti_opts
    else:
        # we keep users items and add default items he has not set
        cfg['alti_opts'] = dict(default_alti_opts.items() + cfg['alti_opts'].items())


def initialization_stats_opts(cfg):
    # class_type can be 'slope' (classification is done from slope) or 'user' (classification from any kind) or None
    # class_rad_range defines the intervals to classify the classification type image from

    default_stats_opts = {'class_type': 'slope',
                          'class_rad_range': [0, 10, 25, 50, 90],
                          'cross_classification': False,
                          'alti_error_threshold': {'value': 0.1, 'unit': 'meter'},
                          'elevation_thresholds' : {'list': [0.5,1,3], 'zunit': 'meter'}}

    if 'stats_opts' not in cfg:
        cfg['stats_opts'] = default_stats_opts
    else:
        # we keep users items and add default items he has not set
        cfg['stats_opts'] = dict(default_stats_opts.items() + cfg['stats_opts'].items())
    try:
        # we try to evaluate what could be set to "True" (or "False) as boolean true (false)
        cfg['stats_opts']['cross_classification'] = ast.literal_eval(cfg['stats_opts']['cross_classification'])
    except ValueError:
        # we assume if previous instruction fails because of a malformed string, then we have the boolean we need
        pass


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


def adjust_tile_size(image_size, tile_size):
    """
    Adjust the size of the tiles.
    """
    tile_w = min(image_size['w'], tile_size)  # tile width
    ntx = int(np.round(float(image_size['w']) / tile_w))
    # ceil so that, if needed, the last tile is slightly smaller
    tile_w = int(np.ceil(float(image_size['w']) / ntx))

    tile_h = min(image_size['h'], tile_size)  # tile height
    nty = int(np.round(float(image_size['h']) / tile_h))
    tile_h = int(np.ceil(float(image_size['h']) / nty))

    print('tile size: {} {}'.format(tile_w, tile_h))

    return tile_w, tile_h


def compute_tiles_coordinates(roi, tile_size_w, tile_size_h):
    """
    """
    out = []
    for r in np.arange(roi['y'], roi['y'] + roi['h'], tile_size_h):
        h = min(tile_size_h, roi['y'] + roi['h'] - r)
        for c in np.arange(roi['x'], roi['x'] + roi['w'], tile_size_w):
            w = min(tile_size_w, roi['x'] + roi['w'] - c)
            out.append((c, r, w, h))

    return out


def divide_images(cfg):
    """
    List the tiles to process and prepare their output directories structures.

    Returns:
        a list of dictionaries. Each dictionary contains the image coordinates
        and the output directory path of a tile.
    """

    # compute biggest roi
    dem = A3DDEMRaster(cfg['inputDSM']['path'],
                       load_data=(cfg['roi'] if 'roi' in cfg else False))

    sizes = {'w': dem.nx, 'h': dem.ny}
    roi = {'x': cfg['roi']['x'] if 'roi' in cfg else 0,
           'y': cfg['roi']['y'] if 'roi' in cfg else 0,
           'w': dem.nx, 'h': dem.ny}

    # list tiles coordinates
    tile_size_w, tile_size_h = adjust_tile_size(sizes, cfg['tile_size'])
    tiles_coords = compute_tiles_coordinates(roi, tile_size_w, tile_size_h)
    if 'max_digit_tile_row' not in cfg:
        cfg['max_digit_tile_row'] = len(str(tiles_coords[len(tiles_coords) - 1][0]))
    if 'max_digit_tile_col' not in cfg:
        cfg['max_digit_tile_col'] = len(str(tiles_coords[len(tiles_coords) - 1][1]))

    # build a tile dictionary for all non-masked tiles and store them in a list
    tiles = []
    for coords in tiles_coords:
        tile = {}
        c, r, w, h = coords
        tile['dir'] = get_tile_dir(cfg, c, r, w, h)
        tile['coordinates'] = coords
        key = str((c, r, w, h))
        tiles.append(tile)

    # make tiles directories and store json configuration
    for tile in tiles:
        mkdir_p(tile['dir'])

        # save a json dump of the tile configuration
        tile_cfg = copy.deepcopy(cfg)
        c, r, w, h = tile['coordinates']
        tile_cfg['roi'] = {'x': c, 'y': r, 'w': w, 'h': h}
        tile_cfg['outputDir'] = tile['dir']

        tile_json = os.path.join(tile['dir'], 'config.json')
        tile['json'] = tile_json

        with open(tile_json, 'w') as f:
            json.dump(tile_cfg, f, indent=2)

    # Write the list of json files to outputDir/tiles.txt
    with open(os.path.join(cfg['outputDir'],'tiles.txt'),'w') as f:
        for tile in tiles:

            f.write(tile['json']+os.linesep)

    return tiles
