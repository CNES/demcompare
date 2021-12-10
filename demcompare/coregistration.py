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
Coregistration part of dsm_compare
"""

# Standard imports
import copy
import os

# Third party imports
import numpy as np

# DEMcompare imports
from .img_tools import save_tif, translate, translate_to_coregistered_geometry
from .nuth_kaab_universal_coregistration import nuth_kaab_lib
from .output_tree_design import get_out_dir, get_out_file_path


def coregister_with_nuth_and_kaab(
    dem, ref, init_disp_x=0, init_disp_y=0, tmp_dir=".", nb_iters=6
):
    """
    Compute x and y offsets between two DEMs
    using Nuth and Kaab (2011) algorithm

    Note that dem will not be resampled in the process
    Note that the dem's georef-origin might be shifted in the process

    :param dem: xarray Dataset, master dem
    :param ref: xarray Dataset, slave dem
    :param init_disp_x: initial x disparity in pixel
    :param init_disp_y: initial y disparity in pixel
    :param tmp_dir: directory path to temporay results (as Nuth & Kaab plots)
    :param nb_iters: Nuth and Kaab number of iterations (default 6)
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

    # We change the georef-origin of nk_a3d_libAPI's coreg DEMs
    # -> this is because NK library takes dem and ref,
    #    and gives back two coreg DEMs keeping the dem's
    #    georef-grid and georef-origin.
    #    This is done by interpolating & resampling the ref.
    #    While this is good behavior for independent use,
    #    it is not exactly what we're looking for.
    #    We do want the ref to be the one resampled,
    #    but we want the coreg DEMs to have the ref's georef-origin .

    # Translate the georef-origin of coreg DEMs based on x_off and y_off values
    #   -> this makes dem coregistered on ref
    #
    # note the -0.5 since the (0,0) pixel coord is pixel centered
    coreg_dem = translate(coreg_dem, x_off - 0.5, -y_off - 0.5)
    coreg_ref = translate(coreg_ref, x_off - 0.5, -y_off - 0.5)
    final_dh = translate(final_dh, x_off - 0.5, -y_off - 0.5)

    # Eventually we return nuth and kaab results :
    #  NB : -y_off because y_off from nk is north oriented
    #       we take into account initial disparity
    return (
        x_off + init_disp_x,
        -y_off + init_disp_y,
        z_off,
        coreg_dem,
        coreg_ref,
        final_dh,
    )


def coregister_and_compute_alti_diff(cfg, dem, ref):
    """
    Coregister two DSMs together
    and compute alti differences (before and after coregistration).
    The two coregistred DSMs are in dem's georef-grid,
    with the ref's georef-origin

    This can be view as a two step process:
    - plani rectification computation
    - alti differences computation

    :param cfg: configuration dictionary
    :param dem: dem raster
    :param ref: reference dem raster
    :return: coreg_dem, coreg_ref and alti differences
    """

    # Get Nuth and Kaab method number of iterations if defined
    if cfg["plani_opts"]["coregistration_iterations"] is not None:
        nb_iters = cfg["plani_opts"]["coregistration_iterations"]
    else:
        # Default to 6 iterations
        nb_iters = 6

    if cfg["plani_opts"]["coregistration_method"] == "nuth_kaab":
        (
            x_bias,
            y_bias,
            z_bias,
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
    cfg["plani_results"] = {}
    cfg["plani_results"]["dx"] = {
        "bias_value": x_bias * coreg_dem.attrs["xres"],
        "unit": coreg_dem.attrs["plani_unit"].name,
    }
    cfg["plani_results"]["dy"] = {
        "bias_value": y_bias * abs(coreg_dem.attrs["yres"]),
        "unit": coreg_dem.attrs["plani_unit"].name,
    }
    # -> for alti_results
    cfg["alti_results"] = {}
    cfg["alti_results"]["rectifiedDSM"] = copy.deepcopy(cfg["inputDSM"])
    cfg["alti_results"]["rectifiedRef"] = copy.deepcopy(cfg["inputRef"])
    cfg["alti_results"]["rectifiedDSM"]["path"] = coreg_dem.attrs["input_img"]
    cfg["alti_results"]["rectifiedRef"]["path"] = coreg_ref.attrs["input_img"]
    cfg["alti_results"]["rectifiedDSM"]["nodata"] = coreg_dem.attrs["no_data"]
    cfg["alti_results"]["rectifiedRef"]["nodata"] = coreg_ref.attrs["no_data"]
    cfg["alti_results"]["dz"] = {
        "bias_value": float(z_bias),
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
