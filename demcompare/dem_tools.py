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
This module contains main functions to manipulate DEM raster images.

It represents the primary API to manipulate DEM as xarray dataset in demcompare.
Dataset and associated internal functions are described in dataset_tools.py
"""

# Standard imports
import copy
import os
from enum import Enum
from typing import Tuple, Union

# Third party imports
import numpy as np
import rasterio
import xarray as xr
from astropy import units as u
from rasterio import Affine

from .dataset_tools import (
    compute_offset_adapting_factor,
    create_dataset,
    reproject_dataset,
)
from .img_tools import convert_pix_to_coord, crop_rasterio_source_with_roi


def load_dem(
    path: str,
    no_data: float = None,
    band: int = 1,
    geoid_georef: bool = False,
    geoid_path: Union[str, None] = None,
    zunit: str = "m",
    input_roi: Union[bool, dict, Tuple] = False,
) -> xr.Dataset:
    """
    Reads the input DEM path and parameters and generates
    the DEM object as xr.Dataset to be handled in demcompare functions.

    A DEM can be any raster file opened by rasterio.

    :param path: path to dem (readable by rasterio)
    :type path: str
    :param no_data: forcing dem no data value
            (None by default and if set inside metadata)
    :type no_data: float or None
    :param band: band to be read in DEM. Default: 1
    :type band: int
    :param geoid_georef: is dem's georef is geoid
    :type geoid_georef: bool
    :param geoid_path: optional path to local geoid, default is EGM96
    :type geoid_path: str or None
    :param zunit: dem z unit
    :type zunit: str
    :param input_roi: False if dem are to be fully loaded,
            other options are a dict roi or Tuple
    :type input_roi: bool, dict or Tuple
    :return: dem  xr.DataSet containing : (see dataset_tools for details)
                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :rtype: xr.Dataset
    """

    # Open dem with rasterio
    # TODO: protect open with use "with" statement to open file
    # but source_rasterio is closed and tests bug
    src_dem = rasterio.open(path)

    # Get rasterio transform :
    #  affine transformation matrix that maps pixel locations
    #    in (row, col) coordinates to (x, y) spatial positions
    dem_geotransform = src_dem.transform
    # Get rasterio BoundingBox(left, bottom, right, top)
    bounds_dem = src_dem.bounds

    # TODO: clarify input_roi parameter
    # TODO: clarify geoid_georef and geoid_path parameters
    if input_roi is not False:
        # Use ROI
        if isinstance(input_roi, dict):
            if (
                "left" in input_roi
                and "bottom" in input_roi
                and "right" in input_roi
                and "top" in input_roi
            ):
                # coordinates
                bounds_dem = rasterio.coords.BoundingBox(
                    input_roi["left"],
                    input_roi["bottom"],
                    input_roi["right"],
                    input_roi["top"],
                )
            elif (
                "x" in input_roi
                and "y" in input_roi
                and "w" in input_roi
                and "h" in input_roi
            ):
                # coordinates
                window_dem = rasterio.windows.Window(
                    input_roi["x"],
                    input_roi["y"],
                    input_roi["w"],
                    input_roi["h"],
                )
                bounds_dem = rasterio.coords.BoundingBox(
                    rasterio.windows.bounds(window_dem, dem_geotransform)
                )

            else:
                raise TypeError("Not the right conventions for ROI")

    # Get dem raster image from band image
    dem_image = src_dem.read(band)

    # create dataset
    dem_dataset = create_dem(
        dem_image,
        dem_geotransform,
        src_dem.crs,
        path,
        bounds_dem,
        no_data=no_data,
        geoid_georef=geoid_georef,
        geoid_path=geoid_path,
        zunit=zunit,
        source_rasterio=src_dem,
    )

    return dem_dataset


def copy_dem(dem: xr.Dataset) -> xr.Dataset:
    """
    Returns a copy of the input dem.

    :param dem: input dem to copy, xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type dem: xr.Dataset
    :return dem_copy: copy of the input dem, xr.DataSet
                containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :rtype: xr.Dataset
    """
    # If present, the source_rasterio has to be temporarily set
    # out of the input dem
    if "source_rasterio" in dem.attrs:
        source_rasterio = dem.attrs["source_rasterio"]
        dem.attrs["source_rasterio"] = None
        dem_copy = copy.deepcopy(dem)
        dem.attrs["source_rasterio"] = source_rasterio
    else:
        dem_copy = copy.deepcopy(dem)

    return dem_copy


def save_dem(
    dataset: xr.Dataset, filename: str, new_array=None, no_data: float = -32768
) -> xr.Dataset:
    """
    Writes a Dataset in a tiff file.
    If new_array is set, new_array is used as data.
    Returns written dataset.

    TODO: remove new_array with stats refacto

    :param dataset:  xarray.DataSet containing the variables :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type dataset: xr.Dataset
    :param filename:  output filename
    :type filename: str
    :param new_array:  new array to write
    :type new_array: np.ndarray or None
    :param no_data:  value of nodata to use
    :type no_data: float
    :return:  xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :rtype: xr.Dataset
    """

    # update from dataset
    previous_profile = {}
    previous_profile["crs"] = dataset.attrs["crs"]
    previous_profile["transform"] = Affine.from_gdal(
        dataset["georef_transform"].data[0],
        dataset["georef_transform"].data[1],
        dataset["georef_transform"].data[2],
        dataset["georef_transform"].data[3],
        dataset["georef_transform"].data[4],
        dataset["georef_transform"].data[5],
    )

    data = dataset["image"].data
    if new_array is not None:
        data = new_array

    if len(dataset["image"].shape) == 2:
        row, col = data.shape
        with rasterio.open(
            filename,
            mode="w+",
            driver="GTiff",
            width=col,
            height=row,
            count=1,
            dtype=data.dtype,
            crs=previous_profile["crs"],
            transform=previous_profile["transform"],
        ) as source_ds:
            source_ds.nodata = no_data
            source_ds.write(data, 1)

    else:
        row, col, depth = data.shape
        with rasterio.open(
            filename,
            mode="w+",
            driver="GTiff",
            width=col,
            height=row,
            count=depth,
            dtype=data.dtype,
            crs=previous_profile["crs"],
            transform=previous_profile["transform"],
        ) as source_ds:
            for dsp in range(1, depth + 1):
                source_ds.write(data[:, :, dsp - 1], dsp)
    dataset.attrs["input_img"] = filename

    return dataset


def translate_dem(
    dataset: xr.Dataset,
    x_offset: Union[float, int, np.ndarray],
    y_offset: Union[float, int, np.ndarray],
) -> xr.Dataset:
    """
    Applies pixellic offset to the input dataset's
    georeference origin by modifying its transform.

    :param dataset:  xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type dataset: xr.Dataset
    :param x_offset: x offset
    :type x_offset: Union[float, int, ndarray]
    :param y_offset: y offset
    :type y_offset: Union[float, int, ndarray]
    :return: translated xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :rtype: xr.Dataset
    """
    dataset_translated = copy.copy(dataset)
    # Project the pixellic offset to input dataset's coordinates
    x_off, y_off = convert_pix_to_coord(
        dataset["georef_transform"].data, y_offset, x_offset
    )
    # To add an offset, the [0] and [3] positions
    # of the transform have to be modified
    dataset_translated["georef_transform"].data[0] = x_off
    dataset_translated["georef_transform"].data[3] = y_off

    return dataset_translated


class SamplingSourceParameter(Enum):
    """
    Enum type definition for sampling_source parameter
    value are dem_to_align or ref to choose in reproject_dems
    """

    DEM_TO_ALIGN = "dem_to_align"
    REF = "ref"


def create_dem(  # pylint: disable=too-many-arguments, too-many-branches
    data: np.ndarray,
    transform: Union[np.ndarray, rasterio.Affine] = None,
    img_crs: Union[rasterio.crs.CRS, None] = None,
    input_img: Union[str, None] = None,
    bounds: rasterio.coords.BoundingBox = None,
    no_data: float = None,
    geoid_georef: bool = False,
    geoid_path: Union[str, None] = None,
    zunit: str = "m",
    source_rasterio: rasterio.DatasetReader = None,
) -> xr.Dataset:
    """
    Creates dem from input array and transform.

    The demcompare DEM is an xarray Dataset containing:
    :image: 2D (row, col) image as xarray.DataArray,
    :georef_transform: georef transform with 6 coefficients

    :param data: image data
    :type data: np.ndarray
    :param transform: rasterio georeferencing transformation matrix
    :type transform: np.ndarray or rasterio.Affine
    :img_crs: image CRS georef
    :type img_crs: Rasterio.crs.CRS
    :param input_img: image path
    :type input_img: str
    :param bounds: dem bounds
    :type bounds: rasterio.coords.BoundingBox or None
    :param no_data: no_data value in the image
    :type no_data: float or None
    :param geoid_georef: if dem's georef is geoid
    :type geoid_georef: bool
    :param geoid_path: optional path to local geoid, default is EGM96
    :type geoid_path: str or None
    :param zunit: unit
    :type zunit: str
    :param source_rasterio: rasterio dataset reader object
    :type source_rasterio: rasterio.DatasetReader or None
    :return: xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :rtype: xr.Dataset
    """

    # If source_rasterio (DatasetReader is given), set img_ds variable
    img_ds = None
    if source_rasterio:
        img_ds = source_rasterio

    # If no no_data value was given
    # If img_ds exists and has nodatavals,
    # Otherwise the default
    # -9999 value is used
    if no_data is None:
        no_data = -9999
        if img_ds:
            meta_nodata = img_ds.nodatavals[0]
            if meta_nodata is not None:
                no_data = meta_nodata

    # If input data has three dimensions, flatten
    if len(data.shape) == 3:
        # to dim 2
        dim_single = np.where(data.shape) == 1
        if dim_single == 0:
            data = data[0, :, :]
        if dim_single == 2:
            data = data[:, :, 0]

    # Convert no_data values to nan
    data = data.astype(np.float32)
    data[data == no_data] = np.nan

    # Convert altimetric units to meter
    data = ((data * u.Unit(zunit)).to(u.meter)).value
    new_zunit = u.meter

    # If no transform was given, add random transform and resolution
    if transform is None:
        xres = 1
        yres = 1
        transform = np.array([0, xres, 0, 0, 0, yres])

    # If input transform is affine, convert it to gdal and np.array
    if isinstance(transform, Affine):
        # to_gdal just switches the input transform parameter's order to:
        # (c, a, b, f, d, e)
        transform = np.array(transform.to_gdal())

    # Obtain image's georef
    # The image's crs is used as georef.
    if not img_crs:
        if img_ds:
            img_crs = img_ds.crs
            # get plani unit
        else:
            # If no dataset image was given,
            # the default "WGS84" georef is considered.
            img_crs = rasterio.crs.CRS.from_epsg(4326)
    if img_crs.is_geographic:
        plani_unit = u.deg
    else:
        plani_unit = u.m

    # If the georef is geoid, set default the geoid_path if not set
    if geoid_georef:
        if geoid_path is None:
            # this returns the fully resolved path to the
            # python installed module
            module_path = os.path.dirname(__file__)
            # Geoid relative Path as installed in setup.py
            geoid_path = "geoid/egm96_15.gtx"
            # Create full geoid path
            geoid_path = os.path.join(module_path, geoid_path)
    else:
        geoid_path = None

    # Create dataset
    dataset = create_dataset(
        data,
        transform,
        img_crs,
        input_img,
        bounds,
        no_data,
        geoid_path,
        plani_unit,
        new_zunit,
        source_rasterio,
    )

    return dataset


def reproject_dems(
    dem_to_align: xr.Dataset,
    ref: xr.Dataset,
    initial_shift_x: Union[int, float] = 0,
    initial_shift_y: Union[int, float] = 0,
    sampling_source: str = SamplingSourceParameter.DEM_TO_ALIGN.value,
    # Disable for unreformable line_too_long
) -> Tuple[xr.Dataset, xr.Dataset, Tuple[float, float]]:
    """
    Reprojects both DEMs to common grid, common bounds and common georef origin.

    The common grid, bounds, georef origin are defined
    by the sampling_source parameter.
    It defines which is the sampling of the output DEMs.

    If sampling_source is "ref":
        ref is cropped to the common bounds
        dem_to_align is cropped to the common bounds and resampled
    If sampling_source is "dem_to_align":
        ref is cropped to the common bounds and resampled
        dem_to_align is cropped to the common bounds

    :param dem_to_align: dem to align xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type dem_to_align: xr.Dataset
    :param ref: ref xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type ref: xr.Dataset
    :param initial_shift_x: optional initial shift x
    :type initial_shift_x: Union[int, float]
    :param initial_shift_y: optional initial shift y
    :type initial_shift_y: Union[int, float]
    :param sampling_source: 'ref' or 'dem_to_align', the sampling
                 value of the output dems, by defaut "dem_to_align"
    :type sampling_source: str
    :return: reproj_cropped_dem_to_align xr.DataSet,
                 reproj_cropped_ref xr.DataSet, adapting_factor.
                 The xr.Datasets containing :

                 - im : 2D (row, col) xarray.DataArray float32
                 - trans: 1D (trans_len) xarray.DataArray
    :rtype: xr.Dataset, xr.Dataset, Tuple[float, float]
    """
    if sampling_source == SamplingSourceParameter.REF.value:
        interp = dem_to_align
        static = ref
        # Compute adapting_factor for later adapting the
        # offset to the original dem resolution
        adapting_factor = compute_offset_adapting_factor(dem_to_align, ref)
    else:  # sampling_source == SamplingSourceParameter.DEM_TO_ALIGN.value:
        interp = ref
        static = dem_to_align
        # If sampling value is dem_to_align, adapting
        # factor is 1 (no adaptation needed)
        adapting_factor = (1.0, 1.0)

        # Get georef and bounds of static
    static_crs = static.attrs["crs"]
    bounds_static = static.attrs["bounds"]

    # Reproject if input image is inversed top bottom or left right
    # to have the same consistent (left, bottom, right, top) reference
    # than interp projected bounds (orientation bug otherwise)
    transformed_static_bounds = rasterio.warp.transform_bounds(
        static_crs,
        static_crs,
        bounds_static[0],
        bounds_static[1],
        bounds_static[2],
        bounds_static[3],
    )

    # Get georef and bounds of interp
    interp_crs = interp.attrs["crs"]
    bounds_interp = interp.attrs["bounds"]

    # Project bounds to static_crs
    transformed_interp_bounds = rasterio.warp.transform_bounds(
        interp_crs,
        static_crs,
        bounds_interp[0],
        bounds_interp[1],
        bounds_interp[2],
        bounds_interp[3],
    )

    # Obtain intersection roi
    if rasterio.coords.disjoint_bounds(
        transformed_static_bounds, transformed_interp_bounds
    ):
        raise NameError("ERROR: ROIs do not intersect")
    intersection_roi = (
        max(transformed_static_bounds[0], transformed_interp_bounds[0]),
        max(transformed_static_bounds[1], transformed_interp_bounds[1]),
        min(transformed_static_bounds[2], transformed_interp_bounds[2]),
        min(transformed_static_bounds[3], transformed_interp_bounds[3]),
    )

    # Crop static dem
    # rasterio.mask.mask function needs the src read by rasterio
    src_static = static.attrs["source_rasterio"]
    (
        new_cropped_static,
        new_cropped_static_transform,
    ) = crop_rasterio_source_with_roi(src_static, intersection_roi)

    # Create cropped static dem
    reproj_cropped_static = create_dem(
        new_cropped_static,
        new_cropped_static_transform,
        img_crs=static_crs,
        no_data=static.attrs["no_data"],
        geoid_path=static.attrs["geoid_path"],
        zunit=static.attrs["zunit"],
        input_img=static.attrs["input_img"],
        bounds=intersection_roi,
    )

    # Full_interp represent a dem with the full interp image
    full_interp = create_dem(
        interp["image"].data,
        interp.georef_transform,
        img_crs=interp_crs,
        no_data=interp.attrs["no_data"],
        geoid_path=interp.attrs["geoid_path"],
        zunit=interp.attrs["zunit"],
        input_img=interp.attrs["input_img"],
        bounds=intersection_roi,
    )

    # Translate dem_to_align according to the initial shift
    if initial_shift_x != 0 or initial_shift_y != 0:
        if sampling_source == SamplingSourceParameter.REF.value:
            # If sampling_source is ref, adapt the initial shift to the new
            # resolution
            x_factor, y_factor = adapting_factor
            full_interp = translate_dem(
                full_interp,
                initial_shift_x * x_factor,
                initial_shift_y * y_factor,
            )
        else:
            reproj_cropped_static = translate_dem(
                reproj_cropped_static, initial_shift_x, initial_shift_y
            )
    # Interp DEM is reprojected into the static DEM's georef-grid
    # Crop and resample are done in the interp DEM
    reproj_cropped_interp = reproject_dataset(
        full_interp, reproj_cropped_static, interp="bilinear"
    )
    # Update dataset input_img with interp old value
    reproj_cropped_interp.attrs["input_img"] = full_interp.attrs["input_img"]

    # Define reprojected ref and dem_to_align according to the sampling value
    if sampling_source == SamplingSourceParameter.REF.value:
        reproj_cropped_ref = reproj_cropped_static
        reproj_cropped_dem_to_align = reproj_cropped_interp
    else:  # sampling_source == SamplingSourceParameter.DEM_TO_ALIGN.value:
        reproj_cropped_ref = reproj_cropped_interp
        reproj_cropped_dem_to_align = reproj_cropped_static

    return reproj_cropped_dem_to_align, reproj_cropped_ref, adapting_factor


def compute_dems_diff(ref: xr.Dataset, dem_to_align: xr.Dataset) -> xr.Dataset:
    """
    Compute altitude difference ref - dem_to_align and
    return it as an xr.Dataset with the dem_to_align
    georeferencing and attributes.

    :param ref: ref xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type ref: xr.Dataset
    :param dem_to_align: dem to alignxr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :type dem_to_align: xr.Dataset
    :return: difference xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :rtype: xr.Dataset
    """
    diff_raster = ref["image"].data - dem_to_align["image"].data
    diff_dem = create_dem(
        diff_raster,
        transform=dem_to_align.georef_transform.data,
        no_data=-32768,
        img_crs=dem_to_align.crs,
    )
    return diff_dem
