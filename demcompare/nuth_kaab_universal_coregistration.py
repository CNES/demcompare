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
Nuth and Kaab universal co-registration
(Correcting elevation data for glacier change detection 2011).

Based on the work of geoutils project
https://github.com/GeoUtils/geoutils/blob/master/geoutils/dem_coregistration.py
Authors : Amaury Dehecq, Andrew Tedstone
Date : June 2015
License : MIT
"""

# Standard imports
import argparse
import os

# Third party imports
import matplotlib.pyplot as pl
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import leastsq

# DEMcompare imports
from .img_tools import load_dems, read_img_from_array, save_tif


def grad2d(dem):
    """

    :param dem:
    :return: slope (fast forward style) and aspect
    """
    grad2, grad1 = np.gradient(dem)  # in Python, x and y axis reversed

    slope = np.sqrt(grad1 ** 2 + grad2 ** 2)
    aspect = np.arctan2(-grad1, grad2)  # aspect=0 when slope facing north
    aspect = aspect + np.pi

    return slope, aspect


def nuth_kaab_single_iter(dh, slope, aspect, plot_file=None):
    """
    Compute the horizontal shift between 2 DEMs
    using the method presented in Nuth & Kaab 2011

    Inputs :
    - dh : array, elevation difference master_dem - slave_dem
    - slope/aspect : array, slope and aspect for the same locations as the dh
    - plot_file : file to where store plot.
        Set to None if plot is to be printed. Set to False for no plot at all.
    Returns :
    - east, north, c : f, estimated easting and northing of the shift,
        c is not used here but is related to the vertical shift
    """
    # The aim is to compute dh / tan(alpha) as a function of the aspect
    # -> hence we are going to be slice the aspect to average a value
    #     for dh / tan(alpha) on those sliced areas
    # -> then we are going to fit the values by the model a.cos(b-aspect)+c
    #    - a will be the magnitude of the horizontal shift
    #    - b will be its orientation
    #    - c will be a vertical mean shift

    # function to be correlated with terrain aspect
    # NB : target = dh / tan(alpha) (see Fig. 2 of Nuth & Kaab 2011)
    # Explicitely ignore divide by zero warning,
    #   as they will be processed as nan later.
    with np.errstate(divide="ignore", invalid="ignore"):
        target = dh / slope
    target = target[np.isfinite(dh)]
    aspect = aspect[np.isfinite(dh)]

    # compute median value of target for different aspect slices
    slice_bounds = np.arange(0, 2 * np.pi, np.pi / 36)
    mean = np.zeros([len(slice_bounds)])
    x_s = np.zeros([len(slice_bounds)])
    j = 0
    for i in slice_bounds:
        target_slice = target[
            (i < aspect) & (aspect < i + np.pi / 36)
        ]  # select target in the slice
        target_slice = target_slice[
            (target_slice < 200) & (target_slice > -200)
        ]  # avoid target>200 and target<-200
        mean[j] = np.median(target_slice)
        x_s[j] = i
        j = j + 1

    # function to fit according to Nuth & Kaab
    x = aspect.ravel()
    y = target.ravel()

    # remove non-finite values
    xf = x[(np.isfinite(x)) & (np.isfinite(y))]
    yf = y[(np.isfinite(x)) & (np.isfinite(y))]

    # remove outliers
    p1 = np.percentile(yf, 1)
    p99 = np.percentile(yf, 99)
    xf = xf[(p1 < yf) & (yf < p99)]
    yf = yf[(p1 < yf) & (yf < p99)]

    # set the first guess
    p0 = (3 * np.std(yf) / (2 ** 0.5), 0, np.mean(yf))

    # least square fit : peval defines the model chosen
    def peval(x, p):
        return p[0] * np.cos(p[1] - x) + p[2]

    def residuals(p, y, x):
        err = peval(x, p) - y
        return err

    # we run the least square fit
    # by minimizing the "distance" between y and peval (see residuals())
    plsq = leastsq(residuals, p0, args=(mean, x_s), full_output=1)
    yfit = peval(x_s, plsq[0])

    # plotting results
    if plot_file is not False:
        pl.figure(1, figsize=(7.0, 8.0))
        pl.plot(x_s, mean, "b.")
        pl.plot(x_s, yfit, "k-")
        # ax.set_ylim([np.min(mean),])
        pl.xlabel("Terrain aspect (rad)")
        pl.ylabel(r"dh/tan($\alpha$)")
        if plot_file:
            pl.savefig(plot_file, dpi=100, bbox_inches="tight")
        else:
            pl.show()
        pl.close()

    a, b, c = plsq[0]
    east = a * np.sin(b)  # with b=0 when north (origin=y-axis)
    north = a * np.cos(b)

    return east, north, c


def nuth_kaab_lib(dsm_dataset, ref_dataset, outdir_plot=None, nb_iters=6):
    """
    This is the lib api of nuth and kaab universal coregistration.
    It offers quite the same services as the classic main api
    but uses Xarray as input instead of raster files.
    This allows the user to pre-process the data
    and / or avoid unecessary reload of the data.

    Output coregister DSM might be saved.
    Plots might be saved as well (and then not printed) if outputPlot is set.

    NB : it is assumed that both dem3Draster have np.nan values inside
    the '.r' field as masked values

    :param dsm_dataset: xarray Dataset
    :param ref_dataset: xarray Dataset
    :param outdir_plot: path to output Plot directory
        (plots are printed if set to None)
    :param nb_iters: default: 6
    :return: x and y shifts (as 'dsm_dataset + (x,y) = ref_dataset')
    """

    # Set target dsm grid for interpolation purpose
    xgrid = np.arange(dsm_dataset["im"].data.shape[1])
    ygrid = np.arange(dsm_dataset["im"].data.shape[0])
    # TODO : not used, to clean
    # x_pixels, y_pixels = np.meshgrid(xgrid, ygrid)
    # trans = dsm_dataset["trans"].data
    # Xgeo = trans[0] + (x_pixels + 0.5) * trans[1]
    #       + (y_pixels + 0.5) * trans[2]
    # Ygeo = trans[3] + (x_pixels + 0.5) * trans[4]
    #       + (y_pixels + 0.5) * trans[5]

    initial_dh = ref_dataset["im"].data - dsm_dataset["im"].data
    coreg_ref = ref_dataset["im"].data

    # Display
    median = np.median(initial_dh[np.isfinite(initial_dh)])
    nmad_old = 1.4826 * np.median(
        np.abs(initial_dh[np.isfinite(initial_dh)] - median)
    )
    maxval = 3 * nmad_old
    pl.figure(1, figsize=(7.0, 8.0))
    pl.imshow(initial_dh, vmin=-maxval, vmax=maxval)
    color_bar = pl.colorbar()
    color_bar.set_label("Elevation difference (m)")
    if outdir_plot:
        pl.savefig(
            os.path.join(outdir_plot, "ElevationDiff_BeforeCoreg.png"),
            dpi=100,
            bbox_inches="tight",
        )
    else:
        pl.show()
    pl.close()

    # Since later interpolations will consider nodata values as normal values,
    # we need to keep track of nodata values
    # to get rid of them when the time comes
    nan_maskval = np.isnan(coreg_ref)
    dsm_from_filled = np.where(nan_maskval, -9999, coreg_ref)

    # Create spline function for interpolation
    spline_1 = RectBivariateSpline(ygrid, xgrid, dsm_from_filled, kx=1, ky=1)
    spline_2 = RectBivariateSpline(ygrid, xgrid, nan_maskval, kx=1, ky=1)
    xoff, yoff, zoff = 0, 0, 0

    print("Nuth & Kaab iterations...")
    coreg_dsm = dsm_dataset["im"].data
    for i in range(nb_iters):
        # remove bias
        coreg_ref -= median

        # Elevation difference
        dh = coreg_dsm - coreg_ref
        slope, aspect = grad2d(coreg_dsm)

        # compute offset
        if outdir_plot:
            plotfile = os.path.join(
                outdir_plot, "nuth_kaab_iter#{}.png".format(i)
            )
        else:
            plotfile = None
        east, north, z = nuth_kaab_single_iter(
            dh, slope, aspect, plot_file=plotfile
        )
        print(
            (
                "#{} - Offset in pixels : ({},{}), -bias : ({})".format(
                    i + 1, east, north, z
                )
            )
        )
        xoff += east
        yoff += north
        zoff += z

        # resample slave DEM in the new grid
        # spline 1 : positive y shift moves south
        znew = spline_1(ygrid - yoff, xgrid + xoff)
        nanval_new = spline_2(ygrid - yoff, xgrid + xoff)

        # we created nan_maskval so that non nan values are set to 0.
        # interpolation "creates" values
        # and the one not affected by nan are the one still equal to 0.
        # hence, all other values must be considered invalid ones
        znew[nanval_new != 0] = np.nan

        # update DEM
        if xoff >= 0:
            coreg_ref = znew[:, 0 : znew.shape[1] - int(np.ceil(xoff))]
            coreg_dsm = dsm_dataset["im"].data[
                :, 0 : dsm_dataset["im"].data.shape[1] - int(np.ceil(xoff))
            ]
        else:
            coreg_ref = znew[:, int(np.floor(-xoff)) : znew.shape[1]]
            coreg_dsm = dsm_dataset["im"].data[
                :, int(np.floor(-xoff)) : dsm_dataset["im"].data.shape[1]
            ]
        if -yoff >= 0:
            coreg_ref = coreg_ref[0 : znew.shape[0] - int(np.ceil(-yoff)), :]
            coreg_dsm = coreg_dsm[
                0 : dsm_dataset["im"].data.shape[0] - int(np.ceil(-yoff)), :
            ]
        else:
            coreg_ref = coreg_ref[int(np.floor(yoff)) : znew.shape[0], :]
            coreg_dsm = coreg_dsm[
                int(np.floor(yoff)) : dsm_dataset["im"].data.shape[0], :
            ]

        # print some statistics
        diff = coreg_ref - coreg_dsm
        diff = diff[np.isfinite(diff)]
        nmad_new = 1.4826 * np.median(np.abs(diff - np.median(diff)))
        median = np.median(diff)

        print(
            (
                "Median : {0:.2f}, NMAD = {1:.2f}, Gain : {2:.2f}".format(
                    median, nmad_new, (nmad_new - nmad_old) / nmad_old * 100
                )
            )
        )
        nmad_old = nmad_new

    print(("Final Offset in pixels (east, north) : ({},{})".format(xoff, yoff)))

    #
    # Get geo raster from coreg_ref array
    #
    coreg_dsm_dataset = read_img_from_array(
        coreg_dsm, from_dataset=dsm_dataset, no_data=-32768
    )
    coreg_ref_dataset = read_img_from_array(
        coreg_ref, from_dataset=dsm_dataset, no_data=-32768
    )
    initial_dh_dataset = read_img_from_array(
        initial_dh, from_dataset=dsm_dataset, no_data=-32768
    )
    final_dh_dataset = read_img_from_array(
        coreg_ref - coreg_dsm, from_dataset=dsm_dataset, no_data=-32768
    )

    return (
        xoff,
        yoff,
        zoff,
        coreg_dsm_dataset,
        coreg_ref_dataset,
        initial_dh_dataset,
        final_dh_dataset,
    )


def run(
    dsm_to,
    dsm_from,
    outfile=None,
    nb_iters=6,
    outdir_plot=None,
    nan_dsm_to=None,
    nan_dsm_from=None,
    save_diff=False,
):
    """
    Coregister dsm_from to dsm_to using Nuth & Kaab (2011) algorithm.

    Output coregister DSM might be saved.
    Plots might be saved as well (and then not printed) if outputPlot is set.

    If nan_dsm and/or nan_ref are not set,
    no data values are read from dsms metadata

    Both input dsm are projected on the same grid :
     - with dsm_to resolution
     - on the biggest common footprint

    TODO: not used in code, Refactor with nuth_kaab_lib function.

    :param dsm_to: path to dsm to coregister to
    :param dsm_from: path to dsm to coregister from
    :param outfile: path to dsm_from after coregistration to dsm_to
    :param nb_iters:
    :param outdir_plot: path to output Plot directory
        (plots are printed if set to None)
    :param nan_dsm_to:
    :param nan_dsm_from:
    :param save_diff: save ./initial_dh.tiff and ./final_dh.tiff
        with dsms diff before and after coregistration
    :return: x and y shifts (as 'dsm_from + (x,y) = dsm_to')
    """

    #
    # Create datasets
    #
    reproj_dem, reproj_ref = load_dems(
        dsm_to,
        dsm_from,
        ref_nodata=nan_dsm_from,
        dem_nodata=nan_dsm_to,
        load_data=False,
    )

    #
    # Coregister and compute diff
    #
    (
        xoff,
        yoff,
        zoff,
        coreg_dsm_dataset,
        coreg_ref_dataset,
        init_dh_dataset,
        final_dh_dataset,
    ) = nuth_kaab_lib(
        reproj_dem, reproj_ref, nb_iters=nb_iters, outdir_plot=outdir_plot
    )

    #
    # Save coreg dem
    #
    if outfile:
        save_tif(coreg_dsm_dataset, "./coreg_dsm.tif")
        save_tif(coreg_ref_dataset, "./coreg_ref.tif")

    #
    # Save diffs
    #
    if save_diff:
        save_tif(init_dh_dataset, "./initial_dh.tif")
        save_tif(final_dh_dataset, "./final_dh.tif")

    return xoff, yoff, zoff


def get_parser():
    """
    Parser of nuth kaab independent main
    TODO: To clean with main. Keep independent main ?
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Universal co-registration method "
        "presented in Nuth & Kaab 2011."
        "NB : 1) It is supposed that both dsms share common reference"
        "            (whether it is geoid or ellipsoid)."
        "     2) DSMs must be georefenced.",
    )

    parser.add_argument("dsm_to", type=str, help="master dsm")
    parser.add_argument(
        "dsm_from", type=str, help="slave dsm you wish to coregister to dsm_to"
    )
    parser.add_argument(
        "-outfile", action="store_true", help="saves output coregistered DSM"
    )
    parser.add_argument(
        "-nb_iters",
        dest="nb_iters",
        type=int,
        default=6,
        help="number of iterations",
    )
    parser.add_argument(
        "-dirplot",
        dest="plot",
        type=str,
        default=None,
        help="path to output plot directory. "
        "Plots are printed if set to None (default)",
    )
    parser.add_argument(
        "-nodata1",
        dest="nodata1",
        type=str,
        default=None,
        help="no data value for DSM to compare "
        "(default value is read in metadata)",
    )
    parser.add_argument(
        "-nodata2",
        dest="nodata2",
        type=str,
        default=None,
        help="no data value for Reference DSM "
        "(default value is read in metadata)",
    )
    parser.add_argument(
        "-save_diff",
        action="store_true",
        help="store on file system a ./initial_dh.tiff and a ./final_dh.tiff "
        "with dsms differences before and after coregistration",
    )

    return parser


def main():
    """
    Main from Nuth Kaab API lib
    """
    parser = get_parser()
    args = parser.parse_args()

    run(
        args.dsm_to,
        args.dsm_from,
        outfile=args.outfile,
        nb_iters=args.nb_iters,
        outdir_plot=args.plot,
        nan_dsm_to=args.nodata1,
        nan_dsm_from=args.nodata2,
        save_diff=args.save_diff,
    )


if __name__ == "__main__":
    main()
