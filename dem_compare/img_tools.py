#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of DEM_COMPARE
#
#     https://github.com/CNES/Pandora_pandora
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
This module contains functions associated to raster images.
"""

import logging
import numpy as np
import rasterio
import rasterio.warp
import rasterio.windows
import rasterio.mask
import rasterio.crs
import xarray as xr
import copy
import os
from astropy import units as u
from affine import Affine
from scipy.ndimage import filters
from rasterio.warp import reproject, Resampling
from scipy import interpolate
try:
    import pyproj
except ImportError:
    import mpl_toolkits.basemap.pyproj as pyproj


def read_image(path, band=1):
    """
    Read image as array

    :param path: path
    :type path: str
    :type band: numero of band to extract
    :type band: int
    :param col: column
    :type col: int
    :return: band
    :rtype: np.ndarray
    """
    img_ds = rasterio.open(path)
    data = img_ds.read(band)
    return data


def pix_to_coord(transform_array, row: int, col: int):
    """
    Transform pixels to coordinates

    :param transform_array: Transform
    :type transform_array: List or np.ndarray
    :type row: row
    :type row: int or np.ndarray
    :param col: column
    :type col: int or np.ndarray
    :return: x,y
    :rtype: Tuple(float, float)
    """
    transform = Affine.from_gdal(transform_array[0], transform_array[1], transform_array[2],
                                 transform_array[3], transform_array[4], transform_array[5])

    x, y = rasterio.transform.xy(transform, row, col, offset='center')

    if type(x) is not int:
        x = np.array(x)
        y = np.array(y)

    return x, y


def reproject_dataset(dataset, from_dataset, interp='bilinear'):
    """
    Reproject dataset, and return the corresponding xarray.DataSet

    :param dataset: Dataset to reproject
    :type dataset: xarray.DataSet
    :type from_dataset: Dataset to get projection from
    :type from_dataset: xarray.DataSet
    :param interp: interpolation method
    :type interp: string
    :return: xarray.DataSet
    :rtype: xarray.DataSet
    """

    # Copy dataset
    reprojected_dataset = copy.copy(from_dataset)

    interpolation_method = Resampling.bilinear
    if interp == 'bilinear':
        interpolation_method = Resampling.bilinear
    elif interp == 'nearest':
        interpolation_method = Resampling.nearest
    else:
        logging.warning("Interpolation method not available, use default 'bilinear'")

    src_transform = Affine.from_gdal(dataset['trans'].data[0], dataset['trans'].data[1], dataset['trans'].data[2],
                                     dataset['trans'].data[3], dataset['trans'].data[4], dataset['trans'].data[5])
    dst_transform = Affine.from_gdal(from_dataset['trans'].data[0], from_dataset['trans'].data[1],
                                     from_dataset['trans'].data[2], from_dataset['trans'].data[3],
                                     from_dataset['trans'].data[4], from_dataset['trans'].data[5])

    source_array = dataset['im'].data
    dest_array = np.zeros_like(from_dataset['im'].data)

    src_crs = rasterio.crs.CRS.from_dict(dataset.attrs['georef'])
    dst_crs = rasterio.crs.CRS.from_dict(from_dataset.attrs['georef'])

    # reproject
    reproject(
        source=source_array,
        destination=dest_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=interpolation_method
    )

    # change output dataset
    reprojected_dataset['im'].data = dest_array
    reprojected_dataset.attrs['no_data'] = dataset.attrs['no_data']
    reprojected_dataset.coords['no_data'].data = dataset.coords['no_data'].data

    return reprojected_dataset


def read_img(img: str, no_data: float = None, ref: str = 'WGS84', zunit: str = 'm',
             load_data: bool = False) -> xr.Dataset:
    """
    Read image and trasform and return the corresponding xarray.DataSet

    :param img: Path to the image
    :type img: string
    :param no_data: no_data value in the image
    :type no_data: float
    :param ref: WGS84 or egm96
    :type ref: str
    :param zunit: unit
    :type zunit: str
    :param load_data: load as dem
    :type load_data: bool
    :return: xarray.DataSet
    :rtype:
        xarray.DataSet containing the variables :
            - im : 2D (row, col) xarray.DataArray float32
            - trans 1D xarray.DataArray float32
    """
    img_ds = rasterio.open(img)
    data = img_ds.read(1)
    transform = img_ds.transform

    dataset = create_dataset(data, transform, img, no_data=no_data, ref=ref, zunit=zunit, load_data=load_data)

    return dataset


def create_dataset(data: np.ndarray, transform: np.ndarray, img: str,  no_data: float = None, ref:  str = 'WGS84',
                   zunit: str = 'm', load_data: bool = False) -> xr.Dataset:
    """
    Create dataset from array and transform, and return the corresponding xarray.DataSet

    :param data: image data
    :type data: np.ndarray
    :param transform: image data
    :type transform: np.ndarray
    :type img: image path
    :type img: str
    :type no_data: no_data value in the image
    :type no_data: float
    :param ref: WGS84 or egm96
    :type ref: str
    :type zunit: unit
    :type zunit: str
    :param load_data: load as dem
    :type load_data: bool
    :return: xarray.DataSet
    :return: xarray.DataSet
    :rtype:
        xarray.DataSet containing the variables :
            - im : 2D (row, col) xarray.DataArray float32
    """

    img_ds = rasterio.open(img)
    georef = img_ds.crs

    # Manage nodata
    if no_data is None:
        meta_nodata = img_ds.nodatavals[0]
        if meta_nodata is not None:
            no_data = meta_nodata
        else:
            no_data = -9999

    if len(data.shape) == 3:
        # to dim 2
        dim_single = np.where(data.shape) == 1
        if dim_single == 0:
            data = data[0, :, :]
        if dim_single == 2:
            data = data[:, :, 0]
    data = data.astype(np.float32)
    data[data == no_data] = np.nan

    # convert to meter
    data = ((data * u.Unit(zunit)).to(u.meter)).value
    new_zunit = u.meter

    dataset = xr.Dataset({'im': (['row', 'col'], data.astype(np.float32))},
                         coords={'row': np.arange(data.shape[0]),
                                 'col': np.arange(data.shape[1])})

    transform = np.array(transform.to_gdal())
    # Add transform
    trans_len = np.arange(0, len(transform))
    dataset.coords["trans_len"] = trans_len
    dataset['trans'] = xr.DataArray(data=transform, dims=['trans_len'])

    # add nodata
    dataset.coords["no_data"] = no_data
    # add zunit
    dataset.coords["zunit"] = new_zunit

    # add plani unit
    if georef.is_geographic:
        plani_unit = u.deg
    else:
        plani_unit = u.m
    dataset.coords["plani_unit"] = plani_unit

    # add resolution
    dataset.coords["xres"] = transform[1]
    dataset.coords["yres"] = transform[5]

    # Add image conf to the image dataset
    dataset.attrs = {'no_data': no_data,
                     'input_img': img,
                     'georef': georef}

    if load_data is not False:
        if ref == 'EGM96':
            # transform to ellipsoid
            egm96_offset = get_egm96_offset(dataset)
            dataset['im'].data += egm96_offset

    return dataset


def read_img_from_array(img_array: np.ndarray, from_dataset: xr.Dataset = None, no_data: float = None) -> xr.Dataset:
    """
    Read image and mask, and return the corresponding xarray.DataSet

    :param img_array: array
    :type img_array: np.ndarray
    :type no_data: no_data value in the image
    :type no_data: float
    :type from_dataset: dataset to copy
    :type from_dataset: xarray.DataSet
    :return: xarray.DataSet
    :return: xarray.DataSet
    :rtype:
        xarray.DataSet containing the variables :
            - im : 2D (row, col) xarray.DataArray float32
    """

    data = np.copy(img_array)

    # Manage nodata
    if no_data is None:
        no_data = -9999

    data[data == no_data] = np.nan

    if from_dataset is None:
        dataset = xr.Dataset({'im': (['row', 'col'], data.astype(np.float32))},
                             coords={'row': np.arange(data.shape[0]),
                                     'col': np.arange(data.shape[1])})

        # add random resolution
        dataset.coords["xres"] = 1
        dataset.coords["yres"] = 1

        # add nodata
        dataset.coords["no_data"] = no_data
        dataset.attrs = {'no_data': no_data}
    else:
        dataset = copy.deepcopy(from_dataset)
        dataset['im'] = None
        dataset.coords['row'] = np.arange(img_array.shape[0])
        dataset.coords['col'] = np.arange(img_array.shape[1])
        dataset['im'] = xr.DataArray(data=img_array.astype(np.float32), dims=['row', 'col'])

    return dataset


def load_dems(ref_path, dem_path, ref_nodata=None, dem_nodata=None,
              ref_georef='WGS84', dem_georef='WGS84', ref_zunit='m', dem_zunit='m', load_data=True):
    """
    Loads both DEMs

    :param ref_path:  path to ref dem
    :type ref_path: str
    :param dem_path:path to sec dem
    :type ref_path: str
    :param ref_nodata: ref no data value (None by default and if set inside metadata)
    :type ref_path: float
    :param dem_nodata: dem no data value (None by default and if set inside metadata)
    :type ref_path: float
    :param ref_georef: ref georef (either WGS84 -default- or EGM96)
    :type ref_path: str
    :param dem_georef: dem georef (either WGS84 -default- or EGM96)
    :type ref_path: str
    :param ref_zunit: ref z unit
    :type ref_path: unit
    :param dem_zunit: dem z unit
    :type ref_path: unit
    :param load_data: True if dem are to be fully loaded, other options are False or a dict roi
    :type ref_path: bool, dict
    :return: ref and dem datasets
    :rtype: Tuple(xarray.Dataset, xarray.Dataset)
    """

    # Get roi of dem

    src_dem = rasterio.open(dem_path)
    dem_crs = src_dem.crs

    dem_trans = src_dem.transform
    bounds_dem = src_dem.bounds

    if load_data is not True:
        # Use ROI
        if isinstance(load_data, tuple) or isinstance(load_data, list):
            bounds_dem = load_data

        elif isinstance(load_data, dict):
            if "left" in load_data and "bottom" in load_data and "right" in load_data and "top" in load_data:
                # coordinates
                bounds_dem = (load_data["left"], load_data["bottom"], load_data["right"], load_data["top"])
            elif "x" in load_data and "y" in load_data and "w" in load_data and "h" in load_data:
                # coordinates
                window_dem = rasterio.windows.Window(load_data["x"], load_data["y"], load_data["w"], load_data["h"])
                bounds_dem = rasterio.windows.bounds(window_dem, dem_trans)
                print(bounds_dem)

            else:
                print("Not he right conventions for ROI")

    # Get roi of ref

    src_ref = rasterio.open(ref_path)
    ref_crs = src_ref.crs

    bounds_ref = src_ref.bounds

    transformed_ref_bounds = rasterio.warp.transform_bounds(ref_crs, dem_crs, bounds_ref[0], bounds_ref[1],
                                                            bounds_ref[2], bounds_ref[3])

    # intersect roi
    if rasterio.coords.disjoint_bounds(bounds_dem, transformed_ref_bounds):
        raise NameError('ERROR: ROIs do not intersect')
    intersection_roi = (max(bounds_dem[0], transformed_ref_bounds[0]),
                        max(bounds_dem[1], transformed_ref_bounds[1]),
                        min(bounds_dem[2], transformed_ref_bounds[2]),
                        min(bounds_dem[3], transformed_ref_bounds[3]))

    # get  crop
    polygon_roi = bounding_box_to_polygone(intersection_roi[0], intersection_roi[1], intersection_roi[2],
                                           intersection_roi[3])
    geom_like_polygon = {"type": "Polygon", "coordinates": [polygon_roi]}

    # crop dem
    new_cropped_dem, new_cropped_dem_transform = rasterio.mask.mask(src_dem, [geom_like_polygon], all_touched=True,
                                                                    crop=True)

    # create datasets

    dem = create_dataset(new_cropped_dem, new_cropped_dem_transform, dem_path, no_data=dem_nodata, ref=dem_georef,
                         zunit=dem_zunit, load_data=load_data)

    # full_ref represent a dataset with the full image
    full_ref = create_dataset(src_ref.read(1), src_ref.transform, ref_path, no_data=ref_nodata, ref=ref_georef,
                              zunit=ref_zunit, load_data=load_data)

    # reproject, crop, resample
    ref = reproject_dataset(full_ref, dem, interp='bilinear')

    return ref, dem


def bounding_box_to_polygone(left, bottom, right, top):
    """
    Transform bounding box to polygon

    :param left: left bound
    :type left: float
    :param bottom: bottom bound
    :type bottom: float
    :param right: right bound
    :type right: float
    :param top: top bound
    :type top: float
    :return: polygon
    :rtype: List
    """

    polygon = [[left, bottom],
               [right, bottom],
               [right, top],
               [left, top],
               [left, bottom]]

    return polygon


def translate(dataset, x_offset, y_offset):
    """
    Modify transform from dataset
    :param dataset:
    :type dataset: xarray Dataset
    :param x_offset: x offset
    :type x_offset: float
    :param y_offset: y offset
    :type y_offset: float
    :return: xarray Dataset
    """
    dataset_translated = copy.copy(dataset)

    x_off, y_off = pix_to_coord(dataset['trans'].data, y_offset, x_offset)
    dataset_translated['trans'].data[0] = x_off
    dataset_translated['trans'].data[3] = y_off

    return dataset_translated


def save_tif(dataset: xr.Dataset, filename: str, new_array=None, no_data=-32768) -> xr.Dataset:
    """
    Write a xarray.Dataset in a tiff file

    :param dataset: dataset
    :type dataset: xarray.Dataset
    :param filename:  output filename
    :type filename: string
    :param new_array:  new array to write
    :type new_array: np.nadarray
    :param no_data:  value of nodata to use
    :type no_data: float
    :return: None
    """

    # update from dataset
    previous_profile = {}
    previous_profile['crs'] = rasterio.crs.CRS.from_dict(dataset.attrs['georef'])
    previous_profile['transform'] = Affine.from_gdal(dataset['trans'].data[0], dataset['trans'].data[1],
                                                     dataset['trans'].data[2], dataset['trans'].data[3],
                                                     dataset['trans'].data[4], dataset['trans'].data[5])

    data = dataset['im'].data
    if new_array is not None:
        data = new_array

    if len(dataset['im'].shape) == 2:
        row, col = data.shape
        with rasterio.open(filename, mode='w+',
                           driver='GTiff',
                           width=col,
                           height=row,
                           count=1,
                           dtype=data.dtype,
                           crs=previous_profile['crs'],
                           transform=previous_profile['transform']) as source_ds:
            source_ds.nodata = no_data
            source_ds.write(data, 1)

    else:
        row, col, depth = data.shape
        with rasterio.open(filename,
                           mode='w+',
                           driver='GTiff',
                           width=col, height=row,
                           count=depth,
                           dtype=data.dtype,
                           crs=previous_profile['crs'],
                           transform=previous_profile['transform']) as source_ds:
            for dsp in range(1, depth + 1):
                source_ds.write(data[:, :, dsp - 1], dsp)

    new_dataset = copy.deepcopy(dataset)
    new_dataset.attrs['ds_file'] = filename

    return new_dataset


def get_slope(dataset, degree=False):
    """
    Compute slope from dataset
    Slope is presented here :
    http://pro.arcgis.com/fr/pro-app/tool-reference/spatial-analyst/how-aspect-works.htm

    :param dataset: dataset
    :type dataset: xarray.Dataset
    :param degree:  True if is in degree
    :type degree: bool
    :return: slope
    :rtype: np.ndarray
    """

    def get_orthodromic_distance(lon1, lat1, lon2, lat2):
        """
        Get Orthodromic distance from two (lat,lon) coordinates

        :param lon1:
        :param lat1:
        :param lon2:
        :param lat2:
        :return: orthodromic distance
        """
        # WGS-84 equatorial radius in km
        re = 6378137.0
        return re * np.arccos(
            np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * np.cos((lon2 - lon1) * np.pi / 180) + np.sin(
                lat1 * np.pi / 180) * np.sin(lat2 * np.pi / 180))

    crs = rasterio.crs.CRS.from_dict(dataset.attrs['georef'])
    if not crs.is_projected:
        # Our dem is not projected, we can't simply use the pixel resolution
        # -> we need to compute resolution between each point
        ny, nx = dataset['im'].data.shape
        xp = np.arange(nx)
        yp = np.arange(ny)
        xp, yp = np.meshgrid(xp, yp)
        lon, lat = pix_to_coord(dataset['trans'].data, yp, xp)
        lonr = np.roll(lon, 1, 1)
        latl = np.roll(lat, 1, 0)

        distx = get_orthodromic_distance(lon, lat, lonr, lat)
        disty = get_orthodromic_distance(lon, lat, lon, latl)

        # deal withs ingularities at edges
        distx[:, 0] = distx[:, 1]
        disty[0] = disty[1]
    else:
        distx = np.abs(dataset.coords["xres"].data)
        disty = np.abs(dataset.coords["yres"].data)

    conv_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    conv_y = conv_x.transpose()

    # Now we do the convolutions :
    gx = filters.convolve(dataset['im'].data, conv_x, mode='reflect')
    gy = filters.convolve(dataset['im'].data, conv_y, mode='reflect')

    # And eventually we do compute tan(slope) and aspect
    tan_slope = np.sqrt((gx / distx) ** 2 + (gy / disty) ** 2) / 8
    slope = np.arctan(tan_slope)

    # Just simple unit change as required
    if degree is False:
        slope *= 100
    else:
        slope = (slope * 180) / np.pi

    return slope


def interpolate_geoid(geoid_filename, coords, interpol_method='linear'):
    """
    Bilinear interpolation of geoid

    :param geoid_filename :  coord geoid_filename
    :type geoid_filename : str
    :param coords :  coords
    :type coords : np array 2xN [lon,lat]
    :param interpol_method  :  interpolation type
    :type interpol_method : str
    :return interpolated position  : [lon,lat,estimate geoid] (3D np.array)
    """
    dataset = rasterio.open(geoid_filename)

    transform = dataset.transform
    step_x = transform[0]
    # ascending
    step_y = - transform[4]
    # 0 or 0.5

    # coin BG
    [ori_x, ori_y] = transform * (0.5, dataset.height - 0.5)  # positions au centre pixel

    last_x = ori_x + step_x * dataset.width
    last_y = ori_y + step_y * dataset.height
    # transform dep to positions
    geoid_values = dataset.read(1)[::-1, :].transpose()
    x = np.arange(ori_x, last_x, step_x)
    # lat must be in ascending order,
    y = np.arange(ori_y, last_y, step_y)
    geoid_grid_coordinates = (x, y)
    interp_geoid = interpolate.interpn(geoid_grid_coordinates, geoid_values, coords, method=interpol_method,
                                       bounds_error=False, fill_value=None)
    return interp_geoid


def get_egm96_offset(dataset):
    """
    Get offset from geoid to ellipsoid

    :param dataset :  dataset
    :type dataset : xarray.Dataset
    :return offset as array
    :rtype np.ndarray
    """

    # Get Geoid path
    # this returns the fully resolved path to the python installed module
    module_path = os.path.dirname(__file__)
    # Geoid relative Path as installed in setup.py
    geoid_path = 'geoid/egm96_15.gtx'
    # Create full geoid path
    geoid_path = os.path.join(module_path, geoid_path)

    ny, nx = dataset['im'].data.shape
    xp = np.arange(nx)
    yp = np.arange(ny)

    # xp in [-180, 180], yp in [-90, 90]
    xp[xp > 180] = xp[xp > 180] - 360
    xp[xp < 180] = xp[xp < 180] + 360
    yp[yp > 90] = yp[yp > 90] - 180
    yp[yp < 90] = yp[yp < 90] + 180

    xp, yp = np.meshgrid(xp, yp)

    lon, lat = pix_to_coord(dataset['trans'].data, yp, xp)

    src_crs = rasterio.crs.CRS.from_dict(dataset.attrs['georef'])
    if src_crs.is_projected:
        # convert to global coordinates
        proj = pyproj.Proj(src_crs)
        lon, lat = proj(lon, lat, inverse=True)

        # transform to list (2xN)
    lon_1d = np.reshape(lon, (dataset['im'].data.shape[0] * dataset['im'].data.shape[1]))
    lat_1d = np.reshape(lat, (dataset['im'].data.shape[0] * dataset['im'].data.shape[1]))
    coords = np.zeros((lon_1d.size, 2))
    coords[:, 0] = lon_1d
    coords[:, 1] = lat_1d

    # Get geoid values
    interp_geoid = interpolate_geoid(geoid_path, coords, interpol_method='linear')

    # transform to array of shape dataset['im'].data.shape
    arr_offset = np.reshape(interp_geoid, dataset['im'].data.shape)

    return arr_offset
