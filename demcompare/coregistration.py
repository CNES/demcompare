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
    dem1, dem2, init_disp_x=0, init_disp_y=0, tmp_dir="."
):
    """
    Compute x and y offsets between two DEMs
    using Nuth and Kaab (2011) algorithm

    Note that dem1 will not be resampled in the process
    Note that dem1 geo reference might be shifted in the process

    :param dem1: xarray Dataset, master dem
    :param dem2: xarray Dataset, slave dem
    :param init_disp_x: initial x disparity in pixel
    :param init_disp_y: initial y disparity in pixel
    :param tmp_dir: directory path to temporay results (as Nuth & Kaab plots)
    :return: mean shifts (x and y), and coregistered DEMs
    """

    # Resample images to pre-coregistered geometry according to the initial disp
    if init_disp_x != 0 or init_disp_y != 0:

        dem1, dem2 = translate_to_coregistered_geometry(
            dem1, dem2, init_disp_x, init_disp_y
        )

    # Compute nuth and kaab coregistration
    # TODO : check init_dh coherence in demcompare code (saved in __init__.py)
    (
        x_off,
        y_off,
        z_off,
        coreg_dem1,
        coreg_dem2,
        _,
        final_dh,
    ) = nuth_kaab_lib(dem1, dem2, outdir_plot=tmp_dir)

    # Instead of taking nk_a3d_libAPI results we change their georef
    # -> this is because NK library takes two DEMs georeferenced
    #    and gives back two coreg DEMs keeping the initial georef.
    #    This is done by interpolating & resampling the REF DEM (dem2 here)
    #    The initial georef is the input DSM (dem1) one,
    #    since dem1 and dem2 have supposedly been reprojected onto dem1
    #    so that dem1 was not resampled
    #    While this is good behavior for independent use,
    #    this is not exactly what we 're wishing for
    #    We do want the REF DEM to be the one resampled,
    #    but we want to keep its georef, and so here is what we do
    #    so that coreg dem from NK are not modified,
    #    but their georef now is the one of dem2
    coreg_dem1 = translate(coreg_dem1, x_off - 0.5, -y_off - 0.5)
    coreg_dem2 = translate(coreg_dem2, x_off - 0.5, -y_off - 0.5)
    final_dh = translate(final_dh, x_off - 0.5, -y_off - 0.5)

    # Eventually we return nuth and kaab results :
    #  NB : -y_off because y_off from nk is north oriented
    #       we take into account initial disparity
    return (
        x_off + init_disp_x,
        -y_off + init_disp_y,
        z_off,
        coreg_dem1,
        coreg_dem2,
        final_dh,
    )


def coregister_and_compute_alti_diff(cfg, dem1, dem2):
    """
    Coregister two DSMs together
    and compute alti differences (before and after coregistration).

    This can be view as a two step process:
    - plani rectification computation
    - alti differences computation

    :param cfg: configuration dictionary
    :param dem1: dem raster
    :param dem2: reference dem raster to be coregistered to dem1 raster
    :return: coreg_dem1, coreg_dem2 and alti differences
    """

    if cfg["plani_opts"]["coregistration_method"] == "nuth_kaab":
        (
            x_bias,
            y_bias,
            z_bias,
            coreg_dem1,
            coreg_dem2,
            final_dh,
        ) = coregister_with_nuth_and_kaab(
            dem1,
            dem2,
            init_disp_x=cfg["plani_opts"]["disp_init"]["x"],
            init_disp_y=cfg["plani_opts"]["disp_init"]["y"],
            tmp_dir=os.path.join(
                cfg["outputDir"], get_out_dir("nuth_kaab_tmp_dir")
            ),
        )
        z_bias = np.nanmean(final_dh["im"].data)
    else:
        raise NameError("coregistration method unsupported")

    # Saves output coreg DEM to file system
    coreg_dem1 = save_tif(
        coreg_dem1,
        os.path.join(cfg["outputDir"], get_out_file_path("coreg_DEM.tif")),
    )
    coreg_dem2 = save_tif(
        coreg_dem2,
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
        "bias_value": x_bias * coreg_dem1.attrs["xres"],
        "unit": coreg_dem1.attrs["plani_unit"].name,
    }
    cfg["plani_results"]["dy"] = {
        "bias_value": y_bias * abs(coreg_dem1.attrs["yres"]),
        "unit": coreg_dem1.attrs["plani_unit"].name,
    }
    # -> for alti_results
    cfg["alti_results"] = {}
    cfg["alti_results"]["rectifiedDSM"] = copy.deepcopy(cfg["inputDSM"])
    cfg["alti_results"]["rectifiedRef"] = copy.deepcopy(cfg["inputRef"])
    cfg["alti_results"]["rectifiedDSM"]["path"] = coreg_dem1.attrs["ds_file"]
    cfg["alti_results"]["rectifiedRef"]["path"] = coreg_dem2.attrs["ds_file"]
    cfg["alti_results"]["rectifiedDSM"]["nodata"] = coreg_dem1.attrs["no_data"]
    cfg["alti_results"]["rectifiedRef"]["nodata"] = coreg_dem2.attrs["no_data"]
    cfg["alti_results"]["dz"] = {
        "bias_value": float(z_bias),
        "unit": coreg_dem1.attrs["zunit"].name,
        "percent": 100
        * np.count_nonzero(~np.isnan(final_dh["im"].data))
        / final_dh["im"].data.size,
    }
    cfg["alti_results"]["dzMap"] = {
        "path": final_dh.attrs["ds_file"],
        "zunit": coreg_dem1.attrs["zunit"].name,
        "nodata": final_dh.attrs["no_data"],
        "nb_points": final_dh["im"].data.size,
        "nb_valid_points": np.count_nonzero(~np.isnan(final_dh["im"].data)),
    }
    cfg["alti_results"]["rectifiedDSM"]["nb_points"] = coreg_dem1[
        "im"
    ].data.size
    cfg["alti_results"]["rectifiedRef"]["nb_points"] = coreg_dem2[
        "im"
    ].data.size
    cfg["alti_results"]["rectifiedDSM"]["nb_valid_points"] = np.count_nonzero(
        ~np.isnan(coreg_dem1["im"].data)
    )
    cfg["alti_results"]["rectifiedRef"]["nb_valid_points"] = np.count_nonzero(
        ~np.isnan(coreg_dem2["im"].data)
    )

    # Print report
    print(
        "Plani 2D shift between input dsm ({}) and input ref ({}) is".format(
            cfg["inputDSM"]["path"], cfg["inputRef"]["path"]
        )
    )

    print(
        " -> row : {}".format(
            cfg["plani_results"]["dy"]["bias_value"]
            * coreg_dem1.attrs["plani_unit"]
        )
    )
    print(
        " -> col : {}".format(
            cfg["plani_results"]["dx"]["bias_value"]
            * coreg_dem1.attrs["plani_unit"]
        )
    )
    print("")
    print(
        "Alti shift between coreg dsm ({}) and coreg ref ({}) is".format(
            cfg["alti_results"]["rectifiedDSM"]["path"],
            cfg["alti_results"]["rectifiedRef"]["path"],
        )
    )

    print((" -> alti : {}".format(z_bias * coreg_dem1.attrs["zunit"])))

    return coreg_dem1, coreg_dem2, final_dh
