#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022 Centre National d'Etudes Spatiales (CNES).
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
This module contains generic functions associated to raster images.
It consists mainly on wrappers to rasterio functions.
"""

# Standard imports
import logging
from typing import List, Tuple, Union

# Third party imports
import numpy as np
import rasterio
import rasterio.crs
import rasterio.mask
import rasterio.warp
import rasterio.windows
from rasterio import Affine


def convert_pix_to_coord(
    transform_array: Union[List, np.ndarray],
    row: Union[float, int, np.ndarray],
    col: Union[float, int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert input (row, col) pixels to dataset geographic coordinates
    from affine rasterio transform in upper left convention.
    See: https://gdal.org/tutorials/geotransforms_tut.html

    :param transform_array: Array containing 6 Affine Geo Transform coefficients
    :type transform_array: List or np.ndarray
    :param row: row to convert
    :type row: float, int or np.ndarray
    :param col: column to convert
    :type col: float, int or np.ndarray
    :return: converted x,y in geographic coordinates from affine transform
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Obtain the dataset transform in affine format from coefficients
    transform = Affine.from_gdal(
        transform_array[0],
        transform_array[1],
        transform_array[2],
        transform_array[3],
        transform_array[4],
        transform_array[5],
    )
    # Set the offset to ul (upper left)
    # Transform the input pixels to dataset geographic coordinates
    x, y = rasterio.transform.xy(transform, row, col, offset="ul")

    if not isinstance(x, int):
        x = np.array(x)
        y = np.array(y)

    return x, y


def crop_rasterio_source_with_roi(
    src: rasterio.DatasetReader, roi: List[float]
) -> Tuple[np.ndarray, Affine]:
    """
    Transforms the input Region of Interest to polygon and
    crops the input rasterio source DEM and its transform.
    If the ROI is outside of the input DEM, an exception is raised.

    :param src: input source dataset in rasterio format
    :type src: rasterio.DatasetReader
    :param roi: region of interest to crop
    :type roi: List[float]
    :return: cropped dem and its affine transform
    :rtype: Tuple[np.ndarray, Affine]
    """

    polygon = [
        [roi[0], roi[1]],
        [roi[2], roi[1]],
        [roi[2], roi[3]],
        [roi[0], roi[3]],
        [roi[0], roi[1]],
    ]
    geom_like_polygon = {"type": "Polygon", "coordinates": [polygon]}
    try:
        new_cropped_dem, new_cropped_transform = rasterio.mask.mask(
            src, [geom_like_polygon], all_touched=True, crop=True
        )
    except ValueError as roi_outside_dataset:
        logging.error(
            "Input ROI coordinates outside of the %s DEM scope.",
            src.files[0],
        )
        raise ValueError from roi_outside_dataset

    return new_cropped_dem, new_cropped_transform


def compute_gdal_translate_bounds(
    y_offset: Union[float, int, np.ndarray],
    x_offset: Union[float, int, np.ndarray],
    shape: Tuple[int, int],
    georef_transform: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Obtain the gdal coordinate bounds to apply the translation offsets to
    the DEM to coregister/translate with gdal.

    The offsets can be applied with the command line:
    gdal_translate -a_ullr <ulx> <uly> <lrx> <lry>
    /path_to_original_dem.tif /path_to_coregistered_dem.tif

    :param y_offset: y pixel offset
    :type y_offset: Union[float, int, ndarray]
    :param x_offset: x pixel offset
    :type x_offset: Union[float, int, ndarray]
    :param shape: rasterio tuple containing x size and y size
    :type shape: Tuple[int, int]
    :param georef_transform: Array with 6 Affine Geo Transform coefficients
    :type georef_transform: np.ndarray
    :return: coordinate bounds to apply the offsets
    :rtype: Tuple[float,float,float,float]
    """
    # Read original secondary dem
    ysize, xsize = shape
    # Compute the coordinates of the new bounds
    x_0, y_0 = convert_pix_to_coord(georef_transform, y_offset, x_offset)
    x_1, y_1 = convert_pix_to_coord(
        georef_transform, y_offset + ysize, x_offset + xsize
    )

    return float(x_0), float(y_0), float(x_1), float(y_1)
