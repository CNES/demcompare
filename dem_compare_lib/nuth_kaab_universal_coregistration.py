#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

"""
Nuth and Kaab universal co-registration (Correcting elevation data for glacier change detection 2011).
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import leastsq
from a3d_modules.a3d_georaster import A3DDEMRaster
from a3d_modules.a3d_georaster import A3DGeoRaster


def grad2d(dem):
    """

    :param dem:
    :return: slope (fast forward style) and aspect
    """
    g2, g1 = np.gradient(dem) # in Python, x and y axis reversed

    slope = np.sqrt(g1**2 + g2**2)
    aspect = np.arctan2(-g1,g2)    #aspect=0 when slope facing north
    aspect = aspect+np.pi

    return slope,aspect


def nuth_kaab_single_iter(dh, slope, aspect, plotFile=None):
    """
    Compute the horizontal shift between 2 DEMs using the method presented in Nuth & Kaab 2011

    Inputs :
    - dh : array, elevation difference master_dem - slave_dem
    - slope/aspect : array, slope and aspect for the same locations as the dh
    - plotFile : file to where store plot. Set to None if plot is to be printed. Set to False for no plot at all.
    Returns :
    - east, north, c : f, estimated easting and northing of the shift, c is not used here but is related to the vertical shift
    """

    # The aim is to compute dh / tan(alpha) as a function of the aspect
    # -> hence we are going to be slice the aspect to average a value for dh / tan(alpha) on those sliced areas
    # -> then we are going to fit the values by the model a.cos(b-aspect)+c
    #    - a will be the magnitude of the horizontal shift
    #    - b will be its orientation
    #    - c will be a vertical mean shift

    # function to be correlated with terrain aspect
    # NB : target = dh / tan(alpha) (see Fig. 2 of Nuth & Kaab 2011)
    target = dh/slope
    target = target[np.isfinite(dh)]
    aspect = aspect[np.isfinite(dh)]

    # compute median value of target for different aspect slices
    slice_bounds = np.arange(0, 2*np.pi, np.pi/36)
    mean = np.zeros([len(slice_bounds)])
    x_s = np.zeros([len(slice_bounds)])
    j=0
    for i in slice_bounds:
        target_slice = target[(i<aspect) & (aspect<i+np.pi/36)] #select target in the slice
        target_slice = target_slice[(target_slice<200) & (target_slice>-200)] #avoid target>200 and target<-200
        mean[j] = np.median(target_slice)
        x_s[j] = i
        j=j+1

    # function to fit according to Nuth & Kaab
    x = aspect.ravel()
    y = target.ravel()

    # remove non-finite values
    xf = x[(np.isfinite(x)) & (np.isfinite(y))]
    yf = y[(np.isfinite(x)) & (np.isfinite(y))]

    # remove outliers
    p1 = np.percentile(yf,1)
    p99 = np.percentile(yf,99)
    xf = xf[(p1<yf) & (yf<p99)]
    yf = yf[(p1<yf) & (yf<p99)]

    # set the first guess
    p0 = (3*np.std(yf)/(2**0.5),0,np.mean(yf))

    # least square fit : peval defines the model chosen
    def peval(x,p):
        return p[0]*np.cos(p[1]-x) + p[2]

    def residuals(p,y,x):
        err = peval(x,p)-y
        return err

    # we run the least square fit by minimizing the "distance" between y and peval (see residuals())
    plsq = leastsq(residuals, p0, args = (mean,x_s),full_output = 1)
    yfit = peval(x_s,plsq[0])

    #plotting results
    if plotFile is not False:
        pl.figure(1, figsize=(7.0, 8.0))
        pl.plot(x_s,mean,'b.')
        pl.plot(x_s,yfit,'k-')
        #ax.set_ylim([np.min(mean),])
        pl.xlabel('Terrain aspect (rad)')
        pl.ylabel(r'dh/tan($\alpha$)')
        if plotFile:
            pl.savefig(plotFile, dpi=100, bbox_inches='tight')
        else:
            pl.show()
        pl.close()

    a,b,c = plsq[0]
    east = a*np.sin(b)     #with b=0 when north (origin=y-axis)
    north = a*np.cos(b)

    return east, north, c


def a3D_libAPI(dsm_dem3Draster, ref_dem3Draster, outdirPlot=None, nb_iters=6):
    """
    This is the lib api of nuth and kaab universal coregistration.
    It offers quite the same services as the classic main api but uses A3DDEMRaster as input instead of raster files
    This allows the user to pre-process the data and / or avoid unecessary reload of the data.

    Output coregister DSM might be saved.
    Plots might be saved as well (and then not printed) if outputPlot is set.

    :param dsm_dem3Draster: A3DDEMRaster
    :param dsm_from: path to dsm to coregister from
    :param nb_iters:
    :param outputDirPlot: path to output Plot directory (plots are printed if set to None)
    :param keep_georef: keep georef for the output coreg dsm
    :return: x and y shifts (as 'dsm_dem3Draster + (x,y) = ref_dem3Draster')
    """

    # Set target dsm grid for interpolation purpose
    xgrid = np.arange(dsm_dem3Draster.nx)
    ygrid = np.arange(dsm_dem3Draster.ny)
    Xpixels, Ypixels = np.meshgrid(xgrid, ygrid)
    trans = dsm_dem3Draster.trans
    Xgeo = trans[0] + (Xpixels + 0.5) * trans[1] + (Ypixels + 0.5) * trans[2]
    Ygeo = trans[3] + (Xpixels + 0.5) * trans[4] + (Ypixels + 0.5) * trans[5]

    initial_dh = ref_dem3Draster.r - dsm_dem3Draster.r
    coreg_ref = ref_dem3Draster.r

    # Display
    median = np.median(initial_dh[np.isfinite(initial_dh)])
    NMAD_old = 1.4826 * np.median(np.abs(initial_dh[np.isfinite(initial_dh)] - median))
    maxval = 3 * NMAD_old
    pl.figure(1, figsize=(7.0, 8.0))
    pl.imshow(initial_dh, vmin=-maxval, vmax=maxval)
    cb = pl.colorbar()
    cb.set_label('Elevation difference (m)')
    if outdirPlot:
        pl.savefig(os.path.join(outdirPlot, "ElevationDiff_BeforeCoreg.png"), dpi=100, bbox_inches='tight')
    else:
        pl.show()
    pl.close()

    # Since later interpolations will consider nodata values as normal values we need to keep track of nodata values to
    # get rid of them when the time comes
    nan_maskval = np.isnan(coreg_ref)
    dsm_from_filled = np.where(nan_maskval, -9999, coreg_ref)

    # Create spline function for interpolation
    f = RectBivariateSpline(ygrid, xgrid, dsm_from_filled, kx=1, ky=1)
    f2 = RectBivariateSpline(ygrid, xgrid, nan_maskval, kx=1, ky=1)
    xoff, yoff, zoff = 0, 0, 0

    print("Nuth & Kaab iterations...")
    coreg_dsm = dsm_dem3Draster.r
    for i in range(nb_iters):
        # remove bias
        coreg_ref -= median

        # Elevation difference
        dh = coreg_dsm - coreg_ref
        slope, aspect = grad2d(coreg_dsm)

        # compute offset
        if outdirPlot:
            plotfile = os.path.join(outdirPlot, "nuth_kaab_iter#{}.png".format(i))
        else:
            plotfile = None
        east, north, z = nuth_kaab_single_iter(dh, slope, aspect, plotFile=plotfile)
        print("#{} - Offset in pixels : ({},{}), -bias : ({})".format(i + 1, east, north, z))
        xoff += east
        yoff += north
        zoff += z

        # resample slave DEM in the new grid
        znew = f(ygrid - yoff, xgrid + xoff)  # positive y shift moves south
        nanval_new = f2(ygrid - yoff, xgrid + xoff)

        # we created nan_maskval so that non nan values are set to 0
        # interpolation "creates" values and the one not affected by nan are the one still equal to 0
        # hence, all other values must be considered invalid ones
        znew[nanval_new != 0] = np.nan

        # update DEM
        if xoff >= 0:
            coreg_ref = znew[:, 0:znew.shape[1] - int(np.ceil(xoff))]
            coreg_dsm = dsm_dem3Draster.r[:, 0:dsm_dem3Draster.r.shape[1] - int(np.ceil(xoff))]
        else:
            coreg_ref = znew[:, int(np.floor(-xoff)):znew.shape[1]]
            coreg_dsm = dsm_dem3Draster.r[:, int(np.floor(-xoff)):dsm_dem3Draster.r.shape[1]]
        if -yoff >= 0:
            coreg_ref = coreg_ref[0:znew.shape[0] - int(np.ceil(-yoff)), :]
            coreg_dsm = coreg_dsm[0:dsm_dem3Draster.r.shape[0] - int(np.ceil(-yoff)), :]
        else:
            coreg_ref = coreg_ref[int(np.floor(yoff)):znew.shape[0], :]
            coreg_dsm = coreg_dsm[int(np.floor(yoff)):dsm_dem3Draster.r.shape[0], :]

        # print some statistics
        diff = coreg_ref - coreg_dsm
        diff = diff[np.isfinite(diff)]
        NMAD_new = 1.4826 * np.median(np.abs(diff - np.median(diff)))
        median = np.median(diff)

        print("Median : {0:.2f}, NMAD = {1:.2f}, Gain : {2:.2f}".format(median, NMAD_new, (NMAD_new - NMAD_old) / NMAD_old * 100))
        NMAD_old = NMAD_new

    print("Final Offset in pixels (east, north) : ({},{})".format(xoff, yoff))

    #
    # Get geo raster from coreg_ref array
    #
    coreg_dsm_dem3Draster = A3DDEMRaster.from_raster(coreg_dsm,
                                                     dsm_dem3Draster.trans,
                                                     "{}".format(dsm_dem3Draster.srs.ExportToProj4()),
                                                     nodata=-32768)
    coreg_ref_dem3Draster = A3DDEMRaster.from_raster(coreg_ref,
                                                     dsm_dem3Draster.trans,
                                                     "{}".format(dsm_dem3Draster.srs.ExportToProj4()),
                                                     nodata=-32768)
    initial_dh_3DRaster = A3DGeoRaster.from_raster(initial_dh,
                                                   dsm_dem3Draster.trans,
                                                   "{}".format(dsm_dem3Draster.srs.ExportToProj4()),
                                                   nodata=-32768)
    final_dh_3DRaster = A3DGeoRaster.from_raster(coreg_ref - coreg_dsm,
                                                 dsm_dem3Draster.trans,
                                                 "{}".format(dsm_dem3Draster.srs.ExportToProj4()),
                                                 nodata=-32768)

    return xoff, yoff, zoff, coreg_dsm_dem3Draster, coreg_ref_dem3Draster, initial_dh_3DRaster, final_dh_3DRaster


def main(dsm_to, dsm_from, outfile=None, nb_iters=6, outputDirPlot=None, nan_dsm_to=None, nan_dsm_from=None, save_diff=False):
    """
    Coregister dsm_from to dsm_to using Nuth & Kaab (2011) algorithm.
    
    Output coregister DSM might be saved.
    Plots might be saved as well (and then not printed) if outputPlot is set.
    
    If nan_dsm and/or nan_ref are not set, no data values are read from dsms metadata

    Both input dsm are projected on the same grid :
     - with dsm_to resolution
     - on the biggest common footprint
    
    :param dsm_to: path to dsm to coregister to
    :param dsm_from: path to dsm to coregister from
    :param outfile: path to dsm_from after coregistration to dsm_to
    :param nb_iters: 
    :param outputDirPlot: path to output Plot directory (plots are printed if set to None)
    :param nan_dsm_to: 
    :param nan_dsm_from:
    :param save_diff: save ./initial_dh.tiff and ./final_dh.tiff with dsms diff before and after coregistration
    :return: x and y shifts (as 'dsm_from + (x,y) = dsm_to')
    """

    #
    # Create A3DDEMRaster
    #
    dem = A3DDEMRaster(dsm_to, nodata=nan_dsm_to)
    ref = A3DDEMRaster(dsm_from, nodata=nan_dsm_from)
    nodata1 = dem.ds.GetRasterBand(1).GetNoDataValue()
    nodata2 = ref.ds.GetRasterBand(1).GetNoDataValue()

    #
    # Reproject DSMs
    #
    biggest_common_grid = dem.biggest_common_footprint(ref.ds_file)
    nx = (biggest_common_grid[1] - biggest_common_grid[0]) / dem.xres
    ny = (biggest_common_grid[2] - biggest_common_grid[3]) / dem.yres
    reproj_dem = dem.reproject(dem.srs, int(nx), int(ny), biggest_common_grid[0], biggest_common_grid[3],
                               dem.xres, dem.yres, nodata=nodata1)
    reproj_ref = ref.reproject(dem.srs, int(nx), int(ny), biggest_common_grid[0], biggest_common_grid[3],
                               dem.xres, dem.yres, nodata=nodata2)
    reproj_dem.r[reproj_dem.r == nodata1] = np.nan
    reproj_ref.r[reproj_ref.r == nodata2] = np.nan

    #
    # Coregister and compute diff
    #
    xoff, yoff, zoff, coreg_dsm_dem3Draster, coreg_ref_dem3Draster, init_dh_georaster, final_dh_georaster = \
        a3D_libAPI(reproj_dem, reproj_ref, nb_iters=nb_iters, outdirPlot=outputDirPlot)

    #
    # Save coreg dem
    #
    if outfile:
        coreg_dsm_dem3Draster.save_geotiff('./coreg_dsm.tif')
        coreg_ref_dem3Draster.save_geotiff('./coreg_ref.tif')

    #
    # Save diffs
    #
    if save_diff:
        init_dh_georaster.save_geotiff('./initial_dh.tiff')
        final_dh_georaster.save_geotiff('./final_dh.tiff')

    return xoff, yoff, zoff


def get_parser():
    parser = argparse.ArgumentParser(os.path.basename(__file__), 
                                     description='The universal co-registration method presented in Nuth & Kaab 2011.'
                                                 'NB : 1) It is supposed that both dsms share common reference (whether it is geoid or ellipsoid).'
                                                 '     2) DSMs must be georefenced.')

    parser.add_argument('dsm_to', type=str, help='master dsm')
    parser.add_argument('dsm_from', type=str, help='slave dsm you wish to coregister to dsm_to')
    parser.add_argument('-outfile', action='store_true', help='saves output coregistered DSM')
    parser.add_argument('-nb_iters', dest='nb_iters', type=int, default=6, help='number of iterations')
    parser.add_argument('-dirplot', dest='plot', type=str, default=None, help='path to output plot directory. Plots are printed if set to None (default)')
    parser.add_argument('-nodata1', dest='nodata1', type=str, default=None, help='no data value for DSM to compare (default value is read in metadata)')
    parser.add_argument('-nodata2', dest='nodata2', type=str, default=None, help='no data value for Reference DSM (default value is read in metadata)')
    parser.add_argument('-save_diff', action='store_true', help='store on file system a ./initial_dh.tiff and a ./final_dh.tiff with dsms differences before and after coregistration')

    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    main(args.dsm_to, args.dsm_from,
         outfile=args.outfile,
         nb_iters=args.nb_iters,
         outputDirPlot=args.plot,
         nan_dsm_to=args.nodata1,
         nan_dsm_from=args.nodata2,
         save_diff=args.save_diff)