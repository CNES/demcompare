#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

"""
Translate DSMs

"""

from osgeo import gdal


def translate_to_coregistered_geometry(dem1, dem2, dx, dy, interpolator=gdal.GRA_Bilinear):
    """
    Translate both DSMs to their coregistered geometry.

    Note that :
         a) The dem2 georef is assumed to be the reference
         b) The dem2 shall be the one resampled as it supposedly is the cleaner one.
    Hence, dem1 is only cropped, dem2 is the only one that might be resampled.
    However, as dem2 is the ref, dem1 georef is translated to dem2 georef.

    :param dem1: A3DDEMRaster, master dem
    :param dem2: A3DDEMRaster, slave dem
    :param dx: f, dx value in pixels
    :param dy: f, dy value in pixels
    :param interpolator: gdal interpolator
    :return: coregistered DEM as A3DDEMRasters
    """

    #
    # Translate the georef of dem1 based on dx and dy values
    #   -> this makes dem1 coregistered on dem2
    #
    # note the -0.5 since the (0,0) pixel coord is pixel centered
    dem1 = dem1.geo_translate(dx - 0.5, dy - 0.5, system='pixel')

    #
    # Intersect and reproject both dsms.
    #   -> intersect them to the biggest common grid now that they have been shifted
    #   -> dem1 is then cropped with intersect so that it lies within intersect but is not resampled in the process
    #   -> reproject dem2 to dem1 grid, the intersection grid sampled on dem1 grid
    #
    biggest_common_grid = dem1.biggest_common_footprint(dem2)
    reproj_dem1 = dem1.crop(biggest_common_grid)
    reproj_dem2 = dem2.reproject(reproj_dem1.srs, int(reproj_dem1.nx), int(reproj_dem1.ny),
                                 reproj_dem1.footprint[0], reproj_dem1.footprint[3],
                                 reproj_dem1.xres, reproj_dem1.yres, nodata=dem2.nodata, interp_type=interpolator)

    return reproj_dem1, reproj_dem2