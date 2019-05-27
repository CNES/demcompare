#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Copyright (C) 2017-2018 Centre National d'Etudes Spatiales (CNES)

"""
Coregistration part of dsm_compare

"""

import re
import os
import copy
import numpy as np
from dem_compare_lib.output_tree_design import get_out_dir, get_out_file_path


def coregister_with_Nuth_and_Kaab(dem1, dem2, init_disp_x=0, init_disp_y=0, tmpDir='.'):
    """
    Compute x and y offsets between two DSMs using Nuth and Kaab (2011) algorithm

    Note that dem1 will not be resampled in the process
    Note that dem1 geo reference might be shifted in the process

    :param dem1: A3DDEMRaster, master dem
    :param dem2: A3DDEMRaster, slave dem
    :param init_disp_x: initial x disparity in pixel
    :param init_disp_y: initial y disparity in pixel
    :param tmpDir: directory path to temporay results (as Nuth & Kaab plots)
    :return: mean shifts (x and y), and coregistered DEMs
    """

    # Resample images to pre-coregistered geometry according to the initial disp
    if init_disp_x != 0 or init_disp_y != 0:
        from translation import translate_to_coregistered_geometry
        dem1, dem2 = translate_to_coregistered_geometry(dem1, dem2, init_disp_x, init_disp_y)


    # Compute nuth and kaab coregistration
    from nuth_kaab_universal_coregistration import a3D_libAPI as nk_a3D_libAPI
    x_off, y_off, z_off, coreg_dem1, coreg_dem2, init_dh, final_dh = nk_a3D_libAPI(dem1,
                                                                                   dem2,
                                                                                   outdirPlot=tmpDir)

    # Instead of taking nk_a3d_libAPI results we change their georef
    # -> this is because
    #    NK library takes two DEMs georeferenced and gives back two coreg DEMs keeping the initial georef
    #    This is done by interpolating & resampling the REF DEM (dem2 here)
    #    The initial georef is the input DSM (dem1) one, since dem1 and dem2 have supposedly been reprojected onto dem1
    #    so that dem1 was not resampled
    #    While this is good behavior for independent use, this is not exactly what we 're wishing for
    #    We do want the REF DEM to be the one resampled, but we want to keep its georef, and so here is what we do
    #    so that coreg dem from NK are not modified, but their georef now is the one of dem2
    coreg_dem1 = coreg_dem1.geo_translate(x_off - 0.5, -y_off - 0.5, system='pixel')
    coreg_dem2 = coreg_dem2.geo_translate(x_off - 0.5, -y_off - 0.5, system='pixel')
    final_dh = final_dh.geo_translate(x_off - 0.5, -y_off - 0.5, system='pixel')

    # Eventually we return nuth and kaab results :
    #  NB : -y_off because y_off from nk is north oriented
    #       we take into account initial disparity
    return x_off + init_disp_x, -y_off + init_disp_y, z_off, coreg_dem1, coreg_dem2, final_dh


def coregister_and_compute_alti_diff(cfg, dem1, dem2):
    """
    Coregister two DSMs together and compute alti differences (before and after coregistration).

    This can be view as a two step process:
    - plani rectification computation
    - alti differences computation

    :param cfg: configuration dictionary
    :param dem1: dem raster
    :param dem2: reference dem raster to be coregistered to dem1 raster
    :return: coreg_dem1, coreg_dem2 and alti differences
    """

    if cfg['plani_opts']['coregistration_method'] == 'nuth_kaab':
        x_bias, y_bias, z_bias, coreg_dem1, coreg_dem2, final_dh = \
            coregister_with_Nuth_and_Kaab(dem1, dem2,
                                          init_disp_x=cfg['plani_opts']['disp_init']['x'],
                                          init_disp_y=cfg['plani_opts']['disp_init']['y'],
                                          tmpDir=os.path.join(cfg['outputDir'], get_out_dir('nuth_kaab_tmpDir')))
        z_bias = np.nanmean(final_dh.r)
    else:
        raise NameError("coregistration method unsupported")

    # Saves output coreg DEM to file system
    coreg_dem1.save_geotiff(os.path.join(cfg['outputDir'], get_out_file_path('coreg_DEM.tif')))
    coreg_dem2.save_geotiff(os.path.join(cfg['outputDir'], get_out_file_path('coreg_REF.tif')))
    final_dh.save_geotiff(os.path.join(cfg['outputDir'], get_out_file_path('final_dh.tif')))

    # Update cfg
    # -> for plani_results
    cfg['plani_results'] = {}
    cfg['plani_results']['dx'] = {'bias_value': x_bias * coreg_dem1.xres,
                                  'unit': coreg_dem1.plani_unit.name}
    cfg['plani_results']['dy'] = {'bias_value': y_bias * abs(coreg_dem1.yres),
                                  'unit': coreg_dem1.plani_unit.name}
    # -> for alti_results
    cfg['alti_results'] = {}
    cfg['alti_results']['rectifiedDSM'] = copy.deepcopy(cfg['inputDSM'])
    cfg['alti_results']['rectifiedRef'] = copy.deepcopy(cfg['inputRef'])
    cfg['alti_results']['rectifiedDSM']['path'] = coreg_dem1.ds_file
    cfg['alti_results']['rectifiedRef']['path'] = coreg_dem2.ds_file
    cfg['alti_results']['rectifiedDSM']['nodata'] = coreg_dem1.nodata
    cfg['alti_results']['rectifiedRef']['nodata'] = coreg_dem2.nodata
    cfg['alti_results']['dz'] = {'bias_value': float(z_bias),
                                 'unit': coreg_dem1.zunit.name,
                                 'percent': 100 * np.count_nonzero(~np.isnan(final_dh.r)) / final_dh.r.size}
    cfg['alti_results']['dzMap'] = {'path': final_dh.ds_file,
                                    'zunit': coreg_dem1.zunit.name,
                                    'nodata': final_dh.nodata,
                                    'nb_points': final_dh.r.size,
                                    'nb_valid_points': np.count_nonzero(~np.isnan(final_dh.r))}
    cfg['alti_results']['rectifiedDSM']['nb_points'] = coreg_dem1.r.size
    cfg['alti_results']['rectifiedRef']['nb_points'] = coreg_dem2.r.size
    cfg['alti_results']['rectifiedDSM']['nb_valid_points'] = np.count_nonzero(~np.isnan(coreg_dem1.r))
    cfg['alti_results']['rectifiedRef']['nb_valid_points'] = np.count_nonzero(~np.isnan(coreg_dem2.r))

    # Print report
    print("Plani 2D shift between input dsm ({}) and input ref ({}) is".format(cfg['inputDSM']['path'],
                                                                               cfg['inputRef']['path']))
    print(" -> row : {}".format(cfg['plani_results']['dy']['bias_value'] * coreg_dem1.plani_unit))
    print(" -> col : {}".format(cfg['plani_results']['dx']['bias_value'] * coreg_dem1.plani_unit))
    print('')
    print("Alti shift between coreg dsm ({}) and coreg ref ({}) is".format(cfg['alti_results']['rectifiedDSM']['path'],
                                                                           cfg['alti_results']['rectifiedRef']['path']))
    print(" -> alti : {}".format(z_bias * coreg_dem1.zunit))

    return coreg_dem1, coreg_dem2, final_dh
