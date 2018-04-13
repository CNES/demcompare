#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

"""
Coregistration part of dsm_compare

"""

import re
import os
import copy
import numpy as np
from a3d_modules.a3d_georaster import A3DDEMRaster, A3DGeoRaster


def guess_disp_param_with_PRO_DecMoy(dem1, dem2, disp_init, tmpDir, roi=None):
    """
    Auto guess disp init and disp range between both DSMs using PRO_DecMoy

    :param dem1: A3DDEMRaster, master dem
    :param dem2: A3DDEMRaster, slave dem
    :param disp_init: {'x': , 'y':}, initial disp
    :param tmpDir: dir, to store temporary file
    :param roi: {'x':,'y':,'h':,'w':}, roi defined by first point (x,y) and size (w,h)
    :return: guessed initial disparity and advised disparity range (both of the form {'x':,'y':}
    """
    from oc import runProDecMoy

    # Dec moy does not work well after filtering
    # TODO : find a way to detect frequency irregularities

    #
    # We need to write down the raster to file system for PRO_DecMoy
    #
    dem1.save_geotiff(os.path.join(tmpDir,'dem1_as_im_ref_in_for_PRO_DecMoy.tiff'))
    dem2.save_geotiff(os.path.join(tmpDir,'dem2_as_im_sec_in_for_PRO_DecMoy.tiff'))

    #
    # Set PRO_DecMoy variables
    #
    images = {'ref': dem1.ds_file, 'sec': dem2.ds_file}
    outputs = {'log': os.path.join(tmpDir, 'Cr_PRO_DecMoy.txt'),
               'log_params': os.path.join(tmpDir, 'Cr_PRO_DecMoy_params.txt')}
    disp = disp_init
    runProDecMoy(images, outputs, disp, roi)

    # reads decmoy output log to get back dx and dy
    lines = []
    dx = None
    dy = None
    with open(outputs['log'],'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('Resultat'):
            dy, dx = re.findall(r"[-+]?\d+", line)
    if dx is None or dy is None :
        raise NameError('PRO_DecMoy has failed somehow to compute plani shift')

    return {'y':float(dy), 'x':float(dx)}, {'y':4,'x':4}


def coregister_with_PRO_Medicis(dem1, dem2, outDir, init_disp={'x':0,'y':0}, range_disp={'x':4,'y':4}, step={'x':1,'y':1}, roi=None):
    """
    Coregister two A3DDEMRasters (dem2 to dem1) with PRO_Medicis (ZNCC)

    :param dem1: A3DDEMRaster, master dem
    :param dem2: A3DDEMRaster, slave dem
    :param outDir: dir, to store temporary results
    :param init_disp: {'x':,'y':}, initial disparity
    :param range_disp: {'x':,'y':}, range disparity
    :param step: {'x':,'y':}, step (PRO_Medicis is a dense correlator)
    :param roi: {'x':,'y':,'h':,'w':}, roi defined by first point (x,y) and size (w,h)
    :return: mean shifts (x and y), and coregistered DEMs
    """

    # Define window size
    win = {'y': 13, 'x': 13}

    # Save dems raster to file system for PRO_Medicis to use them
    dem1.save_geotiff(os.path.join(outDir, 'dem1_as_imaref_for_PRO_Medicis.tiff'))
    dem2.save_geotiff(os.path.join(outDir, 'dem2_as_imasec_for_PRO_Medicis.tiff'))

    # Set PRO_Medicis variables
    images = {'ref':dem1.ds_file, 'sec': dem2.ds_file}
    outputs_med = {'log':os.path.join(outDir, 'Cr_PRO_Medicis.txt'),
                   'log_params': os.path.join(outDir, 'Cr_PRO_Medicis_params.txt'),
                   'res':os.path.join(outDir, 'OutputGrid_Iter1.hdf')}
    if range_disp['x'] == 1:
        range_disp['x'] = 2
        print('Warning : for Medicis a disp_range of 1 is not big enough. Hence 2 will be the value used.')
    if range_disp['y'] == 1:
        range_disp['y'] = 2
        print('Warning : for Medicis a disp_range of 1 is not big enough. Hence 2 will be the value used.')

    # Run PRO_Medicis
    from oc import runProMedicis
    runProMedicis(images,
                  outputs_med,
                  win,
                  init_disp,
                  range_disp,
                  step,
                  roi,
                  mask_value=dem1.ds.GetRasterBand(1).GetNoDataValue())

    # Run PRO_Stats to get mean shifts' values
    outputs_stats = {'cr': os.path.join(outDir, 'Cr_PRO_Stats.txt'),
                     'res': os.path.join(outDir, 'OutputStats_Iter1.txt')}
    from oc import runProStats
    mean, std, percent = runProStats(outputs_med['res'], outputs_stats)

    # Resample images to coregistered geometry
    from translation import translate_to_coregistered_geometry
    coreg_dem1, coreg_dem2 = translate_to_coregistered_geometry(dem1, dem2, mean['x'], mean['y'])

    return mean['x'], mean['y'], coreg_dem1, coreg_dem2


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
    if init_disp_x != 0 and init_disp_y != 0:
        from translation import translate_to_coregistered_geometry
        dem1, dem2 = translate_to_coregistered_geometry(dem1, dem2, init_disp_x, init_disp_y)


    # Compute nuth and kaab coregistration
    from nuth_kaab_universal_coregistration import a3D_libAPI as nk_a3D_libAPI
    x_off, y_off, z_off, coreg_dem1, coreg_dem2, init_dh, final_dh = nk_a3D_libAPI(dem1,
                                                                       dem2,
                                                                       outdirPlot=tmpDir)

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

    # We start with auto guessing initial disparity if required
    if cfg['plani_opts']['auto_guess_disp_params']:
        disp_init, disp_range, = guess_disp_param_with_PRO_DecMoy(dem1, dem2, cfg['plani_opts']['disp_init'],
                                                                  cfg['tmpDir'],(cfg['roi'] if 'roi' in cfg else None))
        cfg['plani_opts']['disp_init'] = disp_init
        cfg['plani_opts']['disp_range'] = disp_range

    if cfg['plani_opts']['coregistration_method'] == 'correlation':
        if cfg['plani_opts']['correlator'] == 'PRO_Medicis':
            x_bias, y_bias, coreg_dem1, coreg_dem2 = coregister_with_PRO_Medicis(dem1, dem2, cfg['tmpDir'],
                                                                                 init_disp=cfg['plani_opts']['disp_init'],
                                                                                 range_disp=cfg['plani_opts']['disp_range'],
                                                                                 roi=(cfg['roi'] if 'roi' in cfg else None))
            #compute Altidiff
            final_dh = A3DGeoRaster.from_raster(coreg_dem1.r - coreg_dem2.r,
                                                coreg_dem1.trans,
                                                "{}".format(coreg_dem1.srs.ExportToProj4()),
                                                nodata=-32768)
            z_bias = np.nanmean(final_dh.r)
        else:
            raise NameError("correlator {} unsupported".format(cfg['plani_opts']['correlator']))
    else:
        if cfg['plani_opts']['coregistration_method'] == 'nuth_kaab':
            x_bias, y_bias, z_bias, coreg_dem1, coreg_dem2, final_dh = coregister_with_Nuth_and_Kaab(dem1, dem2,
                                                                                                     init_disp_x=cfg['plani_opts']['disp_init']['x'],
                                                                                                     init_disp_y=cfg['plani_opts']['disp_init']['y'],
                                                                                                     tmpDir=cfg['tmpDir'])
            z_bias = np.nanmean(final_dh.r)
        else:
            raise NameError("coregistration method unsupported")

    # Saves output coreg DEM to file system
    coreg_dem1.save_geotiff(os.path.join(cfg['outputDir'], 'coreg_DEM.tif'))
    coreg_dem2.save_geotiff(os.path.join(cfg['outputDir'], 'coreg_REF.tif'))
    final_dh.save_geotiff(os.path.join(cfg['outputDir'], 'final_dh.tif'))

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
    print("Plani 2D shift between input dsm ({}) and input ref ({}) is".format(dem1.ds_file,
                                                                               cfg['inputRef']['path']))
    print(" -> row : {}".format(cfg['plani_results']['dy']['bias_value'] * coreg_dem1.plani_unit))
    print(" -> col : {}".format(cfg['plani_results']['dx']['bias_value'] * coreg_dem1.plani_unit))
    print('')
    print("Alti shift between coreg dsm ({}) and coreg ref ({}) is".format(cfg['alti_results']['rectifiedDSM']['path'],
                                                                           cfg['alti_results']['rectifiedRef']['path']))
    print(" -> alti : {}".format(z_bias * coreg_dem1.zunit))

    return coreg_dem1, coreg_dem2, final_dh
