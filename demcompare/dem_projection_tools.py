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
This module contains functions associated to DEM projection.
"""

# Standard imports
import copy
import logging
import os
from typing import List, Tuple, Union

# Third party imports
import numpy as np
import pyproj
import rasterio
import rasterio.crs
import rasterio.mask
import rasterio.warp
import rasterio.windows
import xarray as xr
from rasterio import Affine
from rasterio.warp import Resampling, reproject
from scipy import interpolate
from scipy.ndimage import filters


def _pix_to_coord(
    transform_array: Union[List, np.ndarray],
    row: Union[float, int, np.ndarray],
    col: Union[float, int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform input pixels to dataset coordinates

    :param transform_array: Transform
    :type transform_array: List or np.ndarray
    :param row: row
    :type row: float, int or np.ndarray
    :param col: column
    :type col: float, int or np.ndarray
    :return: x,y
    :rtype: np.ndarray, np.ndarray
    """
    # Obtain the dataset transform in affine format
    transform = Affine.from_gdal(
        transform_array[0],
        transform_array[1],
        transform_array[2],
        transform_array[3],
        transform_array[4],
        transform_array[5],
    )
    # Set the offset to ul (upper left)
    # Transform the input pixels to dataset coordinates
    x, y = rasterio.transform.xy(transform, row, col, offset="ul")

    if not isinstance(x, int):
        x = np.array(x)
        y = np.array(y)

    return x, y


def reproject_dataset(
    dataset: xr.Dataset, from_dataset: xr.Dataset, interp: str = "bilinear"
) -> xr.Dataset:
    """
    Reproject dataset on the from_dataset's georeference origin and grid,
    and return the corresponding xarray.DataSet.
    If no interp is given, default "bilinear" resampling is considered.
    Another available resampling is "nearest".

    :param dataset: Dataset to reproject xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type dataset: xr.Dataset
    :param from_dataset: Dataset to get projection from
                xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type from_dataset: xr.Dataset
    :param interp: interpolation method
    :type interp: str
    :return: reprojected xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :rtype: xr.Dataset
    """

    # Define reprojected dataset
    reprojected_dataset = copy.copy(from_dataset)

    interpolation_method = Resampling.bilinear
    if interp == "bilinear":
        interpolation_method = Resampling.bilinear
    elif interp == "nearest":
        interpolation_method = Resampling.nearest
    else:
        logging.warning(
            "Interpolation method not available, use default 'bilinear'"
        )
    # Get source and destination transforms
    src_transform = Affine.from_gdal(
        dataset["trans"].data[0],
        dataset["trans"].data[1],
        dataset["trans"].data[2],
        dataset["trans"].data[3],
        dataset["trans"].data[4],
        dataset["trans"].data[5],
    )
    dst_transform = Affine.from_gdal(
        from_dataset["trans"].data[0],
        from_dataset["trans"].data[1],
        from_dataset["trans"].data[2],
        from_dataset["trans"].data[3],
        from_dataset["trans"].data[4],
        from_dataset["trans"].data[5],
    )
    # Get source array
    source_array = dataset["im"].data
    # Define dest_array with the output size and fill with nodata
    dest_array = np.zeros_like(from_dataset["im"].data)
    dest_array[:, :] = -9999
    # Obtain datasets CRSs
    src_crs = rasterio.crs.CRS.from_dict(dataset.attrs["crs"])
    dst_crs = rasterio.crs.CRS.from_dict(from_dataset.attrs["crs"])

    # Reproject with rasterio
    reproject(
        source=source_array,
        destination=dest_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=interpolation_method,
        src_nodata=dataset.attrs["no_data"],
        dst_nodata=-9999,
    )

    # Convert output dataset's remaining nodata values to nan
    dest_array[dest_array == -9999] = np.nan
    # Charge reprojected_dataset's data and nodata values
    reprojected_dataset["im"].data = dest_array
    reprojected_dataset.attrs["no_data"] = dataset.attrs["no_data"]

    return reprojected_dataset


def translate_dataset(
    dataset: xr.Dataset,
    x_offset: Union[float, int, np.ndarray],
    y_offset: Union[float, int, np.ndarray],
) -> xr.Dataset:
    """
    Applies pixellic offset to the input dataset's
    georefence origin by modifying its transform.

    :param dataset:  xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type dataset: xr.Dataset
    :param x_offset: x offset
    :type x_offset: Union[float, int, ndarray]
    :param y_offset: y offset
    :type y_offset: Union[float, int, ndarray]
    :return: translated xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :rtype: xr.Dataset
    """
    dataset_translated = copy.copy(dataset)
    # Project the pixellic offset to input dataset's coordinates
    x_off, y_off = _pix_to_coord(dataset["trans"].data, y_offset, x_offset)
    # To add an offset, the [0] and [3] positions
    # of the transform have to be modified
    dataset_translated["trans"].data[0] = x_off
    dataset_translated["trans"].data[3] = y_off

    return dataset_translated


def get_slope(dataset: xr.Dataset, degree: bool = False) -> np.ndarray:
    """
    Computes DEM's slope
    Slope is presented here :
    http://pro.arcgis.com/ \
            fr/pro-app/tool-reference/spatial-analyst/how-aspect-works.htm

    :param dataset: dataset
    :type dataset: xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :param degree:  True if is in degree
    :type degree: bool
    :return: slope
    :rtype: np.ndarray
    """
    # TODO: see if this function is moved with the refacto stats
    def get_orthodromic_distance(
        lon1: Union[float, np.ndarray],
        lat1: Union[float, np.ndarray],
        lon2: Union[float, np.ndarray],
        lat2: Union[float, np.ndarray],
    ):
        """
        Get Orthodromic distance from two (lat,lon) coordinates

        :param lon1: longitude 1
        :type lon1: Union[float, np.ndarray]
        :param lat1: latitude 1
        :type lat1: Union[float, np.ndarray]
        :param lon2: longitude 2
        :type lon2: Union[float, np.ndarray]
        :param lat2: latitude 2
        :type lat2: Union[float, np.ndarray]
        :return: orthodromic distance
        """
        # WGS-84 equatorial radius in km
        radius_equator = 6378137.0
        return radius_equator * np.arccos(
            np.cos(lat1 * np.pi / 180)
            * np.cos(lat2 * np.pi / 180)
            * np.cos((lon2 - lon1) * np.pi / 180)
            + np.sin(lat1 * np.pi / 180) * np.sin(lat2 * np.pi / 180)
        )

    crs = dataset.attrs["crs"]
    if not crs.is_projected:
        # Our dem is not projected, we can't simply use the pixel resolution
        # -> we need to compute resolution between each point
        ny, nx = dataset["im"].data.shape
        xp = np.arange(nx)
        yp = np.arange(ny)
        xp, yp = np.meshgrid(xp, yp)
        lon, lat = _pix_to_coord(dataset["trans"].data, yp, xp)
        lonr = np.roll(lon, 1, 1)
        latl = np.roll(lat, 1, 0)

        distx = get_orthodromic_distance(lon, lat, lonr, lat)
        disty = get_orthodromic_distance(lon, lat, lon, latl)

        # deal withs ingularities at edges
        distx[:, 0] = distx[:, 1]
        disty[0] = disty[1]
    else:
        distx = np.abs(dataset.attrs["xres"])
        disty = np.abs(dataset.attrs["yres"])

    conv_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    conv_y = conv_x.transpose()

    # Now we do the convolutions :
    gx = filters.convolve(dataset["im"].data, conv_x, mode="reflect")
    gy = filters.convolve(dataset["im"].data, conv_y, mode="reflect")

    # And eventually we do compute tan(slope) and aspect
    tan_slope = np.sqrt((gx / distx) ** 2 + (gy / disty) ** 2) / 8
    slope = np.arctan(tan_slope)

    # Just simple unit change as required
    if degree is False:
        slope *= 100
    else:
        slope = (slope * 180) / np.pi

    return slope


def _interpolate_geoid(
    geoid_filename: str, coords: np.ndarray, interpol_method: str = "linear"
) -> np.ndarray:
    """
    Bilinear interpolation of the given geoid to the input coordinates.
    If no interpol_method is given, a "linear" interpolation is considered.
    If the input coordinates are outside of the geoid scope,
    an exception is raised.

    :param geoid_filename: coord geoid_filename
    :type geoid_filename: str
    :param coords: coords matrix 2xN [lon,lat]
    :type coords: np.ndarray
    :param interpol_method: interpolation type
    :type interpol_method: str
    :return: interpolated position [lon,lat,estimate geoid]
    :rtype: 3D np.array
    """
    dataset = rasterio.open(geoid_filename)

    transform = dataset.transform
    # Get transform's step
    step_x = transform[0]
    step_y = -transform[4]

    # coin BG
    [ori_x, ori_y] = transform * (
        0.5,
        dataset.height - 0.5,
    )  # positions au centre pixel

    # Compute last x and y geoid positions
    last_x = ori_x + step_x * dataset.width
    last_y = ori_y + step_y * dataset.height
    # Get all geoid values
    geoid_values = dataset.read(1)[::-1, :].transpose()
    # Get all geoid positions
    x = np.arange(ori_x, last_x, step_x)
    y = np.arange(ori_y, last_y, step_y)
    geoid_grid_coordinates = (x, y)
    # Interpolate geoid on the input coordinates
    interp_geoid = interpolate.interpn(
        geoid_grid_coordinates,
        geoid_values,
        coords,
        method=interpol_method,
        bounds_error=True,
        fill_value=None,
    )

    return interp_geoid


def get_geoid_offset(
    dataset: xr.Dataset, geoid_path: Union[str, None]
) -> np.ndarray:
    """
    Computes the geoid offset of the input DEM. If no geoid_path is
    given, the default geoid/egm96_15.gtx if used.

    :param dataset: xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type dataset: xr.Dataset
    :param geoid_path: optional absolut geoid_path, if None egm96 is used
    :type geoid_path: str or None
    :return: offset as array
    :rtype: np.ndarray
    """
    # If no geoid path has been given, use the default geoid egm96
    # installed in setup.py
    if geoid_path is None:
        # this returns the fully resolved path to the python installed module
        module_path = os.path.dirname(__file__)
        # Geoid relative Path as installed in setup.py
        geoid_path = "geoid/egm96_15.gtx"
        # Create full geoid path
        geoid_path = os.path.join(module_path, geoid_path)

    # Obtain dataset's grid
    ny, nx = dataset["im"].data.shape
    xp = np.arange(nx)
    yp = np.arange(ny)

    xp, yp = np.meshgrid(xp, yp)
    # Project the dataset grid into lat/lon coordinates
    lonlat = list(_pix_to_coord(dataset["trans"].data, yp, xp))

    # If the georef's units are meters (if is_projected),
    # convert them to degrees
    src_crs = rasterio.crs.CRS.from_dict(dataset.attrs["crs"])
    if src_crs.is_projected:
        # convert to global coordinates
        proj = pyproj.Proj(src_crs)
        lonlat = list(proj(lonlat[0], lonlat[1], inverse=True))

    # transform to list (2xN)
    lon_1d = np.reshape(
        lonlat[0], (dataset["im"].data.shape[0] * dataset["im"].data.shape[1])
    )
    lat_1d = np.reshape(
        lonlat[1], (dataset["im"].data.shape[0] * dataset["im"].data.shape[1])
    )
    coords = np.zeros((lon_1d.size, 2))
    coords[:, 0] = lon_1d
    coords[:, 1] = lat_1d
    # Interpolate geoid on the dataset coordinates
    # If the dataset coordinates are outside of the geoid scope,
    # an error will be raised.
    try:
        # Get geoid values
        interp_geoid = _interpolate_geoid(
            geoid_path, coords, interpol_method="linear"
        )
    except ValueError:
        logging.error(
            "Input DSM {} coordinates outside of the"
            " {} geoid scope.".format(dataset.attrs["input_img"], geoid_path)
        )
        raise

    # transform to array of shape dataset['im'].data.shape
    arr_offset = np.reshape(interp_geoid, dataset["im"].data.shape)

    return arr_offset


def compute_adapting_factor(
    dem_to_align: xr.Dataset, ref: xr.Dataset
) -> Tuple[float, float]:
    """
    Compute factor to adapt computed offsets
    to the dem resolution

    :param dem_to_align: dem_to_align
    :type dem_to_align: xr.Dataset
    :param ref: ref
    :type ref: xr.Dataset
    :return: x and y factors
    :rtype: Tuple[float, float]
    """
    # Obtain the original dem size
    orig_ysize, orig_xsize = dem_to_align["im"].shape
    # Reproject the dem to the ref without doing any crop
    reproj_dem_to_align = reproject_dataset(
        dem_to_align, ref, interp="bilinear"
    )
    # Obtain the full reprojected dem size
    reproj_ysize, reproj_xsize = reproj_dem_to_align["im"].shape
    # Return the x and y factors to adapt the computed offsets
    return orig_xsize / reproj_xsize, orig_ysize / reproj_ysize


def compute_offset_bounds(
    y_off: Union[float, int, np.ndarray],
    x_off: Union[float, int, np.ndarray],
    shape: Tuple[int, int],
    georef_transform: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Obtain the coordinate bounds to apply the offsets to
    the secondary DEM with gdal.

    The offsets can be applied with the command line:
    gdal_translate -a_ullr <ulx> <uly> <lrx> <lry>
    /path_to_original_dem.tif /path_to_coregistered_dem.tif

    :param y_off: y pixel offset
    :type y_off: Union[float, int, ndarray]
    :param x_off: x pixel offset
    :type x_off: Union[float, int, ndarray]
    :return: coordinate bounds to apply the offsets
    :rtype: Tuple[float,float,float,float]
    """
    # Read original secondary dem
    ysize, xsize = shape
    # Compute the coordinates of the new bounds
    x_0, y_0 = _pix_to_coord(georef_transform, y_off, x_off)
    x_1, y_1 = _pix_to_coord(georef_transform, y_off + ysize, x_off + xsize)

    return float(x_0), float(y_0), float(x_1), float(y_1)
