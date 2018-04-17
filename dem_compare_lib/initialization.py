#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Copyright (C) 2017-2018 Centre National d'Etudes Spatiales (CNES)

"""
Init part of dsm_compare
This is where high level parameters are checked and default options are set

"""

from osgeo import gdal
from astropy import units as u


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
    cfg['inputDSM']['path'] = str(cfg['inputDSM']['path'])
    cfg['inputRef']['path'] = str(cfg['inputRef']['path'])

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
                          'alti_error_threshold': {'value': 0.1, 'unit': 'meter'}}

    if 'stats_opts' not in cfg:
        cfg['stats_opts'] = default_stats_opts
    else:
        # we keep users items and add default items he has not set
        cfg['stats_opts'] = dict(default_stats_opts.items() + cfg['stats_opts'].items())
