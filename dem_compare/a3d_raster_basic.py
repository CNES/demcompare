#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of dem_compare
#

import numpy as np
from osgeo import gdal, ogr
import logging
from scipy.ndimage import filters
logger = logging.getLogger('default')


class A3DRasterBasic(object):
    # Filepath and name
    ds_file = None
    # GDAL handle to the dataset
    ds = None
    # Footprint of raster
    footprint = None
    # Raster size
    nx = None
    ny = None
    # Numpy array of band data
    r = None
    # Band datatype
    dtype = None
    # No data value
    nodata = None
    # Band
    band = 1

    def __del__(self):
        """
        Close gdal dataset
        """
        self.ds = None

    def __init__(self, image_array,nodata=None):

        # Import band datatype
        self.dtype = image_array.dtype

        self.r = image_array
        self.nx = self.r.shape[0]
        self.ny = self.r.shape[1]

        self.nodata = -32768
        if nodata is not None:
            self.nodata = nodata

        if nodata is not None:
            self.r[self.r == nodata] = np.nan

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._import_ds(self.ds_file)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('ds')
        return state

    @property
    def dataset_file(self):
        """
        :return: dataset file
        """
        return self.ds_file

    @property
    def dataset(self):
        """
        :return: dataset
        """
        # TODO: copy
        return self.ds

    @property
    def size(self):
        """
        :return: tuple with sizex, sizey
        """
        return (self.nx, self.ny)

    @property
    def data_type(self):
        """
        :return: data type for the reference band
        """
        return self.dtype

    @property
    def no_data_value(self):
        """
        :return: no data value
        """
        return self.nodata

    @staticmethod
    def biggest_common_footprint_from_list_a3d_raster(list_of_a3d_raster):
        """

        :param list_of_a3d_raster:
        :return: left, right, bottom, top of the enveloppe of the intersection of all given a3d_raster
        """
        left, right, bottom, top = list_of_a3d_raster[0].footprint
        wkt = 'POLYGON ((%f %f, %f %f, %f %f, %f %f, %f %f))' % (left, bottom, left, top, right, top, right, bottom,
                                                                 left, bottom)
        poly1 = ogr.CreateGeometryFromWkt(wkt)

        for a3d_raster in list_of_a3d_raster[1:]:
            left, right, bottom, top = a3d_raster.footprint
            wkt = 'POLYGON ((%f %f, %f %f, %f %f, %f %f, %f %f))' % (left, bottom, left, top, right, top, right, bottom,
                                                                     left, bottom)
            poly2 = ogr.CreateGeometryFromWkt(wkt)
            intersect = poly1.Intersection(poly2)
            poly1 = intersect

        enveloppe = poly1.GetEnvelope()
        # TODO: ATTENTION COORDONNEES INVERSEES
        return [enveloppe[0], enveloppe[1], enveloppe[3], enveloppe[2]]

    @classmethod
    def from_path(cls, raster, gdal_dtype=gdal.GDT_Float32, nodata=None):
        """
        Create A3DGeoRaster from raster and geo information

        :param raster: 2D numpy array
        :param gdal_dtype:
        :param nodata:
        :return: A3DGeoRaster
        """
        # Open file
        if isinstance(raster, str):
            ds_file = raster
            ds = gdal.Open(raster, 0)
        # Or import GDAL Dataset from memory
        elif isinstance(raster, gdal.Dataset):
            ds = raster
            ds_file = raster.GetDescription()

        ds = gdal.Open(raster, 0)

        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        nodata = band.GetNoDataValue()

        # Return a fully loaded georaster instance instantiated by the GDAL raster
        return cls(arr, nodata=nodata)

    def get_slope_and_aspect(self, degree=False):
        """
        Return slope and aspect numpy arrays

        Slope & Aspects are computed as presented here :
        http://pro.arcgis.com/fr/pro-app/tool-reference/spatial-analyst/how-aspect-works.htm

        :param degree: boolean, set to True if returned slope in degree (False, default value, for percentage)
        :return: dem slope numpy array
        """

        distx = np.abs(1)
        disty = np.abs(1)

        # Prepare convolving filters to compute the slope
        # Note that :
        #       * scipy.signal.convolve does not deal with np.nan
        #       * astropy.convolution.convolve does not deal with filter whose coefficients sum is 0 (which is our case)
        #       * numpy.convolve does not deal with 2D array so we would have to separate our slope filter
        #         and perform two numpy.convolve, that will do it but we eventually found out :
        #       * scipy.ndimages.filters.convolve ! And it looks like it does the job
        f1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) # for "gradient" x
        f2 = f1.transpose()                         # for "gradient" y

        # Now we do the convolutions :
        gx = filters.convolve(self.r, f1, mode='reflect')
        gy = filters.convolve(self.r, f2, mode='reflect')

        # And eventually we do compute tan(slope) and aspect
        tan_slope = np.sqrt((gx/distx)**2 + (gy/disty)**2)/8
        slope = np.arctan(tan_slope)
        aspect = np.arctan2(gy, gx)

        # Just simple unit change as required
        if degree is False:
            slope *= 100
        else :
            slope = (slope*180)/np.pi

        return slope, aspect

    def save_tiff(self, filename, dtype=gdal.GDT_Float32):
        """
        Save georaster into file system.

        :param filename: output filename
        :param dtype: gdal type
        """
        # deal with unicode input
        filename = str(filename)

        # replace nan by nodata before writing
        self.r[np.where(np.isnan(self.r))] = self.nodata

        dst_ds = write_tiff(filename, self.r, dtype=dtype, nodata=self.nodata)

        # replace nodata by nan after writing
        self.r[self.r == self.nodata] = np.nan


def write_tiff(outputFile, raster, mask=None, dtype=gdal.GDT_Float32, nodata=-999):
    """

    :param outfile:
    :param raster: numpy array
    :param mask: mask band
    :param dtype: GDAL type
    :param nodata: nodata value to set
    :return:

    Based on http://adventuresindevelopment.blogspot.com/2008/12/python-gdal-adding-geotiff-meta-data.html
    and http://www.gdal.org/gdal_tutorial.html and https://github.com/atedstone/georaster
    """

    # Check if the image is multi-band or not.
    if raster.shape.__len__() == 2:
        nbands = 1
        ydim = raster.shape[0]
        xdim = raster.shape[1]
    else:
        raise NameError('ERROR: Raster shape can only be two (y, x)')

    # Setup geotiff file.
    if outputFile != '':
        driver = gdal.GetDriverByName("GTiff")
    else:
        driver = gdal.GetDriverByName('MEM')

    dst_ds = driver.Create(outputFile, xdim, ydim, nbands, dtype)
    # Top left x, w-e pixel res, rotation, top left y, rotation, n-s pixel res
    # Write array
    dst_ds.GetRasterBand(1).WriteArray(raster)
    dst_ds.GetRasterBand(1).SetNoDataValue(nodata)
    if mask:
        dst_ds.GetRasterBand(1).GetMaskBand().WriteArray(mask)

    if outputFile != '':
        dst_ds = None
        return True
    else:
        return dst_ds