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
This module contains functions associated to Demcompare's DEM dataset
"""

import copy

# Standard imports
import logging
import os
from typing import Dict, Tuple, Union

# Third party imports
import numpy as np
import pyproj
import rasterio
import rasterio.crs
import rasterio.mask
import rasterio.warp
import rasterio.windows
import xarray as xr
from astropy import units as u
from rasterio import Affine
from rasterio.warp import Resampling, reproject
from scipy import interpolate

# Demcompare imports
from .img_tools import convert_pix_to_coord


def create_dataset(  # pylint: disable=too-many-arguments, too-many-branches
    data: np.ndarray,
    transform: Union[np.ndarray, rasterio.Affine] = None,
    img_crs: Union[rasterio.crs.CRS, None] = None,
    input_img: Union[str, None] = None,
    bounds: rasterio.coords.BoundingBox = None,
    nodata: float = None,
    geoid_path: Union[str, None] = None,
    plani_unit: u = None,
    zunit: str = "m",
    source_rasterio: Dict[str, rasterio.DatasetReader] = None,
    classification_layer_masks: Union[Dict, xr.DataArray] = None,
) -> xr.Dataset:
    """
    Creates dataset from input array and transform,
    and return the corresponding xarray.DataSet.

    The demcompare dataset is an xarray Dataset containing:
    :image: 2D (row, col) image as xarray.DataArray,
    :georef_transform: 1D (trans_len) xarray.DataArray with the parameters:

                - c: x-coordinate of the upper left pixel,
                - a: pixel size in the x-direction in map units/pixel,
                - b: rotation about x-axis,
                - f: y-coordinate of the upper left pixel,
                - d: rotation about y-axis,
                - e: pixel size in the y-direction in map units, negative

    :classification_layer_masks: 3D (row, col, indicator) xarray.DataArray:

                It contains the maps of all classification layers,
                being the indicator a list with each
                classification_layer name.

    :attributes:

                - nodata : image nodata value. float
                - input_img : image input path. str or None
                - crs : image crs. rasterio.crs.CRS
                - xres : x resolution (value of transform[1]). float
                - yres : y resolution (value of transform[5]). float
                - plani_unit : georefence's planimetric unit. astropy.units
                - zunit : input image z unit value. astropy.units
                - bounds : image bounds. rasterio.coords.BoundingBox
                - geoid_path : geoid path. str or None
                - source_rasterio : rasterio's DatasetReader object or None.

    :param data: image data
    :type data: np.ndarray
    :param transform: rasterio georeferencing transformation matrix
    :type transform: np.ndarray or rasterio.Affine
    :param input_img: image path
    :type input_img: str
    :param bounds: dem bounds
    :type bounds: rasterio.coords.BoundingBox or None
    :param nodata: nodata value in the image
    :type nodata: float or None
    :param geoid_path: optional path to local geoid, default is EGM96
    :type geoid_path: str or None
    :param zunit: unit
    :type zunit: str
    :param source_rasterio: rasterio dataset reader object
    :type source_rasterio: Dict[str,rasterio.DatasetReader] or None
    :param classification_layer_masks: classification layers
    :type classification_layer_masks: Dict, xr.DataArray or None
    :return:  xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type data: xr.Dataset
    """

    dataset = xr.Dataset(
        {"image": (["row", "col"], data.astype(np.float32))},
        coords={
            "row": np.arange(data.shape[0]),
            "col": np.arange(data.shape[1]),
        },
    )

    # Add classification layers
    if isinstance(classification_layer_masks, dict):
        # Define coords, the third col is the indicator
        # with the classification layer name
        coords_classification_layers = [
            dataset.coords["row"],
            dataset.coords["col"],
            classification_layer_masks["names"],
        ]
        # Create the dataarray
        dataset["classification_layer_masks"] = xr.DataArray(
            data=classification_layer_masks["map_arrays"],
            coords=coords_classification_layers,
            dims=["row", "col", "indicator"],
        )
    elif isinstance(classification_layer_masks, xr.DataArray):
        dataset["classification_layer_masks"] = classification_layer_masks

    # Add transform to dataset
    trans_len = np.arange(0, len(transform))
    dataset.coords["trans_len"] = trans_len
    dataset["georef_transform"] = xr.DataArray(
        data=transform, dims=["trans_len"]
    )

    # Add image attributes to the image dataset
    dataset.attrs = {
        "nodata": nodata,
        "input_img": input_img,
        "crs": img_crs,
        "xres": transform[1],
        "yres": transform[5],
        "plani_unit": plani_unit,
        "zunit": zunit,
        "bounds": bounds,
        "source_rasterio": source_rasterio,
    }

    # If the georef is geoid, add geoid offset to the data
    if geoid_path:
        # transform to ellipsoid
        geoid_offset = _get_geoid_offset(dataset, geoid_path)
        dataset["image"].data += geoid_offset
        dataset.attrs["geoid_path"] = geoid_path
    else:
        dataset.attrs["geoid_path"] = None

    return dataset


def reproject_dataset(
    dataset: xr.Dataset, from_dataset: xr.Dataset, interp: str = "bilinear"
) -> xr.Dataset:
    """
    Reproject dataset on the from_dataset's georeference origin and grid,
    and return the corresponding xarray.DataSet.
    If no interp is given, default "bilinear" resampling is considered.
    Another available resampling is "nearest".

    :param dataset: Dataset to reproject xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type dataset: xr.Dataset
    :param from_dataset: Dataset to get projection from
                xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type from_dataset: xr.Dataset
    :param interp: interpolation method
    :type interp: str
    :return: reprojected xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :rtype: xr.Dataset
    """

    # Define reprojected dataset
    reprojected_dataset = copy.copy(from_dataset)
    if "indicator" in reprojected_dataset.coords:
        reprojected_dataset = reprojected_dataset.drop_dims("indicator")
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
        dataset["georef_transform"].data[0],
        dataset["georef_transform"].data[1],
        dataset["georef_transform"].data[2],
        dataset["georef_transform"].data[3],
        dataset["georef_transform"].data[4],
        dataset["georef_transform"].data[5],
    )
    dst_transform = Affine.from_gdal(
        from_dataset["georef_transform"].data[0],
        from_dataset["georef_transform"].data[1],
        from_dataset["georef_transform"].data[2],
        from_dataset["georef_transform"].data[3],
        from_dataset["georef_transform"].data[4],
        from_dataset["georef_transform"].data[5],
    )
    # Get source array
    source_array = dataset["image"].data
    # Define dest_array with the output size and fill with nodata
    dest_array = np.zeros_like(from_dataset["image"].data)
    dest_array[:, :] = from_dataset.nodata
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
        src_nodata=dataset.attrs["nodata"],
        dst_nodata=from_dataset.attrs["nodata"],
    )

    # Convert output dataset's remaining nodata values to nan
    dest_array[dest_array == dataset.attrs["nodata"]] = np.nan
    # Charge reprojected_dataset's data and nodata values
    reprojected_dataset["image"].data = dest_array
    reprojected_dataset.attrs["nodata"] = dataset.attrs["nodata"]

    if "indicator" in dataset.coords:
        indicator = (
            dataset["classification_layer_masks"].coords["indicator"].data
        )
        classification_layer_masks = np.full(
            (
                reprojected_dataset["image"].shape[0],
                reprojected_dataset["image"].shape[1],
                len(indicator),
            ),
            np.nan,
            dtype=np.float32,
        )
        for idx in np.arange(len(indicator)):
            # Define dest_array with the output size and fill with nodata
            dest_array_classif = np.zeros_like(from_dataset["image"].data)
            dest_array_classif[:, :] = dataset.nodata
            # Get source array
            source_array_classif = dataset["classification_layer_masks"][
                :, :, idx
            ].data
            # Reproject with rasterio
            reproject(
                source=source_array_classif,
                destination=dest_array_classif,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=interpolation_method,
                src_nodata=dataset.attrs["nodata"],
                dst_nodata=from_dataset.attrs["nodata"],
            )
            classification_layer_masks[
                :, :, idx
            ].data = dest_array_classif  # type: ignore

        # Define coords, the third col is the indicator
        # with the classification layer name
        coords_classification_layers = [
            reprojected_dataset.coords["row"],
            reprojected_dataset.coords["col"],
            indicator,
        ]
        # Create the dataarray
        reprojected_dataset["classification_layer_masks"] = xr.DataArray(
            data=classification_layer_masks,
            coords=coords_classification_layers,
            dims=["row", "col", "indicator"],
        )
    return reprojected_dataset


def compute_offset_adapting_factor(
    sec: xr.Dataset, ref: xr.Dataset
) -> Tuple[float, float]:
    """
    Compute the factor to adapt the coregistration offsets
    to the dem resolution

    The name is too generic to know the usage quickly. Is the function
    in dem_tools or here ?

    :param sec: sec
    :type sec: xr.Dataset
    :param ref: ref
    :type ref: xr.Dataset
    :return: x and y factors
    :rtype: Tuple[float, float]
    """
    # Obtain the original dem size
    orig_ysize, orig_xsize = sec["image"].shape
    # Reproject the dem to the ref without doing any crop
    reproj_sec = reproject_dataset(sec, ref, interp="bilinear")
    # Obtain the full reprojected dem size
    reproj_ysize, reproj_xsize = reproj_sec["image"].shape
    # Return the x and y factors to adapt the computed offsets
    return orig_xsize / reproj_xsize, orig_ysize / reproj_ysize


def _get_geoid_offset(
    dataset: xr.Dataset, geoid_path: Union[str, None]
) -> np.ndarray:
    """
    Computes the geoid offset of the input DEM. If no geoid_path is
    given, the default geoid/egm96_15.gtx if used.

    :param dataset: xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
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
    ny, nx = dataset["image"].data.shape
    xp = np.arange(nx)
    yp = np.arange(ny)

    xp, yp = np.meshgrid(xp, yp)
    # Project the dataset grid into lat/lon coordinates
    lonlat = list(
        convert_pix_to_coord(dataset["georef_transform"].data, yp, xp)
    )

    # If the georef's units are meters (if is_projected),
    # convert them to degrees
    src_crs = rasterio.crs.CRS.from_dict(dataset.attrs["crs"])
    if src_crs.is_projected:
        # convert to global coordinates
        proj = pyproj.Proj(src_crs)
        lonlat = list(proj(lonlat[0], lonlat[1], inverse=True))

    # transform to list (2xN)
    lon_1d = np.reshape(
        lonlat[0],
        (dataset["image"].data.shape[0] * dataset["image"].data.shape[1]),
    )
    lat_1d = np.reshape(
        lonlat[1],
        (dataset["image"].data.shape[0] * dataset["image"].data.shape[1]),
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
            "Input DSM %s coordinates outside of the %s geoid scope.",
            dataset.attrs["input_img"],
            geoid_path,
        )
        raise

    # transform to array of shape dataset['im'].data.shape
    arr_offset = np.reshape(interp_geoid, dataset["image"].data.shape)

    return arr_offset


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
