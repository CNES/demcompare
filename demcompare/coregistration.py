#!/usr/bin/env python
# coding: utf8
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
Coregistration part of dsm_compare
"""

# Standard imports
import copy
import os
from typing import Dict

# Third party imports
import numpy as np
import xarray as xr

# DEMcompare imports
from .img_tools import (
    compute_offset_bounds,
    save_tif,
    translate,
    translate_to_coregistered_geometry,
)
from .nuth_kaab_universal_coregistration import nuth_kaab_lib
from .output_tree_design import get_out_dir, get_out_file_path


def coregister_with_nuth_and_kaab(
    dem: xr.Dataset,
    ref: xr.Dataset,
    init_disp_x: int = 0,
    init_disp_y: int = 0,
    tmp_dir: str = ".",
    nb_iters: int = 6,
):
    """
    Compute x and y offsets between two DEMs
    using Nuth and Kaab (2011) algorithm

    Note that dem will not be resampled in the process
    Note that the dem's georef-origin might be shifted in the process

    :param dem: master dem
    :type dem: demxarray Dataset
    :param ref: xarray Dataset, slave dem
    :type ref: demxarray Dataset
    :param init_disp_x: initial x disparity in pixel
    :type init_disp_x: int
    :param init_disp_y: initial y disparity in pixel
    :type init_disp_y: int
    :param tmp_dir: directory path to temporary results (as Nuth & Kaab plots)
    :type tmp_dir: str
    :param nb_iters: Nuth and Kaab number of iterations (default 6)
    :type nb_iters: int
    :return: mean shifts (x and y), and coregistered DEMs
    """

    # Resample images to pre-coregistered geometry according to the initial disp
    if init_disp_x != 0 or init_disp_y != 0:

        dem, ref = translate_to_coregistered_geometry(
            dem, ref, init_disp_x, init_disp_y
        )

    # Compute nuth and kaab coregistration
    # TODO : check init_dh coherence in demcompare code (saved in __init__.py)
    (
        x_off,
        y_off,
        z_off,
        coreg_dem,
        coreg_ref,
        _,
        final_dh,
    ) = nuth_kaab_lib(dem, ref, outdir_plot=tmp_dir, nb_iters=nb_iters)

    # Change the georef-origin of nk_a3d_libAPI's coreg DEMs
    # Translate the georef-origin of coreg DEMs based on x_off and y_off values
    #   -> this makes both dems be on the same intermediate georef origin
    #
    coreg_dem = translate(coreg_dem, x_off, -y_off)
    coreg_ref = translate(coreg_ref, x_off, -y_off)
    final_dh = translate(final_dh, x_off, -y_off)

    # Eventually we return nuth and kaab results :
    return (
        x_off,
        y_off,
        z_off,
        coreg_dem,
        coreg_ref,
        final_dh,
    )


def coregister_and_compute_alti_diff(
    cfg: Dict, dem: xr.Dataset, ref: xr.Dataset
):
    """
    Coregister two DSMs together
    and compute alti differences (before and after coregistration).
    The two coregistred DSMs are in dem's georef-grid,
    with the ref's georef-origin

    This can be view as a two step process:
    - plani rectification computation
    - alti differences computation

    :param cfg: configuration dictionary
    :type cfg: Dict
    :param dem: dem raster
    :type dem: xr.Dataset
    :param ref: reference dem raster
    :type ref: xr.Dataset
    :return: coreg_dem, coreg_ref and alti differences
    :rtype: xr.Dataset, xr.Dataset, Dict
    """

    # Get Nuth and Kaab method number of iterations if defined
    if cfg["plani_opts"]["coregistration_iterations"] is not None:
        nb_iters = cfg["plani_opts"]["coregistration_iterations"]
    else:
        # Default to 6 iterations
        nb_iters = 6

    if cfg["plani_opts"]["coregistration_method"] == "nuth_kaab":
        (
            dx_nuth,
            dy_nuth,
            dz_nuth,  # pylint:disable=unused-variable
            coreg_dem,
            coreg_ref,
            final_dh,
        ) = coregister_with_nuth_and_kaab(
            dem,
            ref,
            init_disp_x=cfg["plani_opts"]["disp_init"]["x"],
            init_disp_y=cfg["plani_opts"]["disp_init"]["y"],
            tmp_dir=os.path.join(
                cfg["outputDir"], get_out_dir("nuth_kaab_tmp_dir")
            ),
            nb_iters=nb_iters,
        )
        z_bias = np.nanmean(final_dh["im"].data)
    else:
        raise NameError("coregistration method unsupported")

    # Saves output coreg DEM to file system
    coreg_dem = save_tif(
        coreg_dem,
        os.path.join(cfg["outputDir"], get_out_file_path("coreg_DEM.tif")),
    )
    coreg_ref = save_tif(
        coreg_ref,
        os.path.join(cfg["outputDir"], get_out_file_path("coreg_REF.tif")),
    )
    final_dh = save_tif(
        final_dh,
        os.path.join(cfg["outputDir"], get_out_file_path("final_dh.tif")),
    )

    # Update cfg
    # -> for plani_results
    #  NB : -y_off because y_off from nk is north oriented
    #       we take into account initial disparity
    dx_bias = (dx_nuth + cfg["plani_opts"]["disp_init"]["x"]) * coreg_dem.attrs[
        "xres"
    ]
    dy_bias = (-dy_nuth + cfg["plani_opts"]["disp_init"]["y"]) * abs(
        coreg_dem.attrs["yres"]
    )
    cfg["plani_results"] = {}
    cfg["plani_results"]["dx"] = {
        "nuth_offset": round(dx_nuth, 5),
        "unit_nuth_offset": "px",
        "bias_value": round(dx_bias, 5),
        "unit_bias_value": coreg_dem.attrs["plani_unit"].name,
    }
    cfg["plani_results"]["dy"] = {
        "nuth_offset": round(dy_nuth, 5),
        "unit_nuth_offset": "px",
        "bias_value": round(dy_bias, 5),
        "unit_bias_value": coreg_dem.attrs["plani_unit"].name,
    }

    # -> for the coordinate bounds to apply the offsets
    #    to the original DSM with GDAL
    ulx, uly, lrx, lry = compute_offset_bounds(-dy_nuth, dx_nuth, cfg)
    cfg["plani_results"]["gdal_translate_bounds"] = {
        "ulx": round(ulx, 5),
        "uly": round(uly, 5),
        "lrx": round(lrx, 5),
        "lry": round(lry, 5),
    }

    # -> for alti_results
    cfg["alti_results"] = {}
    cfg["alti_results"]["rectifiedDSM"] = copy.deepcopy(cfg["inputDSM"])
    cfg["alti_results"]["rectifiedRef"] = copy.deepcopy(cfg["inputRef"])
    cfg["alti_results"]["rectifiedDSM"]["path"] = coreg_dem.attrs["input_img"]
    cfg["alti_results"]["rectifiedRef"]["path"] = coreg_ref.attrs["input_img"]
    cfg["alti_results"]["rectifiedDSM"]["nodata"] = coreg_dem.attrs["no_data"]
    cfg["alti_results"]["rectifiedRef"]["nodata"] = coreg_ref.attrs["no_data"]
    dz_bias = float(z_bias)
    cfg["alti_results"]["dz"] = {
        "bias_value": round(dz_bias, 5),
        "unit": coreg_dem.attrs["zunit"].name,
        "percent": 100
        * np.count_nonzero(~np.isnan(final_dh["im"].data))
        / final_dh["im"].data.size,
    }
    cfg["alti_results"]["dzMap"] = {
        "path": final_dh.attrs["input_img"],
        "zunit": coreg_dem.attrs["zunit"].name,
        "nodata": final_dh.attrs["no_data"],
        "nb_points": final_dh["im"].data.size,
        "nb_valid_points": np.count_nonzero(~np.isnan(final_dh["im"].data)),
    }
    cfg["alti_results"]["rectifiedDSM"]["nb_points"] = coreg_dem["im"].data.size
    cfg["alti_results"]["rectifiedRef"]["nb_points"] = coreg_ref["im"].data.size
    cfg["alti_results"]["rectifiedDSM"]["nb_valid_points"] = np.count_nonzero(
        ~np.isnan(coreg_dem["im"].data)
    )
    cfg["alti_results"]["rectifiedRef"]["nb_valid_points"] = np.count_nonzero(
        ~np.isnan(coreg_ref["im"].data)
    )

    # Print report
    print("# Coregistration results:")
    print("\nPlanimetry 2D shift between DEM and REF:")
    print(
        " -> row : {}".format(
            cfg["plani_results"]["dy"]["bias_value"]
            * coreg_dem.attrs["plani_unit"]
        )
    )
    print(
        " -> col : {}".format(
            cfg["plani_results"]["dx"]["bias_value"]
            * coreg_dem.attrs["plani_unit"]
        )
    )
    print("DEM: {}".format(cfg["inputDSM"]["path"]))
    print("REF: {}".format(cfg["inputRef"]["path"]))
    print("\nAltimetry shift between COREG_DEM and COREG_REF")
    print((" -> alti : {}".format(z_bias * coreg_dem.attrs["zunit"])))

    return coreg_dem, coreg_ref, final_dh
