#!/usr/bin/env python
# coding: utf8
# TODO: suppress after unnecesary functions are deleted
# pylint:disable=too-many-lines
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
This module contains functions associated to loading raster images.
"""

import copy

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
import xarray as xr
from astropy import units as u
from rasterio import Affine

from .dem_projection_tools import (
    get_geoid_offset,
    reproject_dataset,
    translate_dataset,
)


def create_dataset_from_dataset(
    img_array: np.ndarray,
    from_dataset: xr.Dataset = None,
    no_data: float = None,
) -> xr.Dataset:
    """
    Creates dataset with img_array data and from_dataset's
    attributes and georefence origin.
    If from_dataset is None defaults attributes are set.

    :param img_array: array
    :type img_array: np.ndarray
    :param from_dataset: dataset to copy
    :type from_dataset: xr.Dataset or None
    :param no_data: no_data value in the image
    :type no_data: float or None
    :return:  xr.Dataset containing

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :rtype: xr.Dataset
    """

    data = np.copy(img_array)
    # If from_dataset is None, use default attributes
    if from_dataset is None:
        # Manage nodata
        if no_data is None:
            no_data = -9999
        data[data == no_data] = np.nan
        # Add random transform and resolution
        xres = 1
        yres = 1
        transform = np.array([0, xres, 0, 0, 0, yres])
        # Create dataset with input data and default attributes
        dataset = create_dataset(data, transform)

        # add nodata
        dataset.attrs["no_data"] = no_data
    else:
        # If present, suppress source_rasterio attribute
        # to enable the deepcopy
        if from_dataset.attrs["source_rasterio"]:
            from_dataset.attrs["source_rasterio"] = None
        dataset = copy.deepcopy(from_dataset)
        # Manage nodata
        if no_data is None:
            no_data = -9999
        else:
            # If no_data is given, the dataset's nodata is overwriten
            dataset.attrs["no_data"] = no_data
        data[data == no_data] = np.nan

        dataset["im"] = None
        dataset.coords["row"] = np.arange(data.shape[0])
        dataset.coords["col"] = np.arange(data.shape[1])
        # Create dataset with input data and from_dataset attributes
        dataset["im"] = xr.DataArray(
            data=data.astype(np.float32), dims=["row", "col"]
        )

    return dataset


def read_image(path: str, band: int = 1) -> np.ndarray:
    """
    Read image as array with optional band value

    :param path: path
    :type path: str
    :param band: numero of band to extract
    :type band: int
    :return: band array
    :rtype: np.ndarray
    """
    # TODO: see if this function is kept after
    # the stats refactoring
    img_ds = rasterio.open(path)
    data = img_ds.read(band)
    return data


def create_dataset(
    data: np.ndarray,
    transform: Union[np.ndarray, None] = None,
    img_crs: Union[rasterio.crs.CRS, None] = None,
    input_img: Union[str, None] = None,
    bounds: Union[np.ndarray, Tuple[int, int, int, int]] = None,
    no_data: float = None,
    geoid_georef: bool = False,
    geoid_path: Union[str, None] = None,
    zunit: str = "m",
    source_rasterio: rasterio.DatasetReader = None,
) -> xr.Dataset:
    """
    Creates dataset from input array and transform,
    and return the corresponding xarray.DataSet.

    The demcompare dataset is an xarray Dataset containing:
    :im: 2D (row, col) xarray.DataArray,
    :trans: 1D (trans_len) xarray.DataArray,
    :attributes:

                - no_data : image nodata value. float
                - input_img : image input path. str or None
                - georef : image georeference. str
                - xres : x resolution (value of transform[1]). float
                - yres : y resolution (value of transform[5]). float
                - plani_unit : georefence's planimetric unit
                  ('deg' or 'meter'). str
                - zunit : image z unit value ('deg' or 'meter'). str
                - bounds : image bounds. Tuple(float, float, float, float)
                  or rasterio.BoundingBox
                - geoid_georef : if the georefence is geoid. bool
                - geoid_path : geoid path (only considered
                  if geoid_georef is True). str or None
                - source_rasterio : rasterio's DatasetReader object or None.

    :param data: image data
    :type data: np.ndarray
    :param transform: image data
    :type transform: np.ndarray
    :param input_img: image path
    :type input_img: str
    :param bounds: dem bounds
    :type bounds: np ndarray or Tuple[int, int, int, int]
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
    :return:  xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type data: xr.Dataset
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

    # Convert units to meter
    data = ((data * u.Unit(zunit)).to(u.meter)).value
    new_zunit = u.meter

    # Create xr.Dataset with data
    dataset = xr.Dataset(
        {"im": (["row", "col"], data.astype(np.float32))},
        coords={
            "row": np.arange(data.shape[0]),
            "col": np.arange(data.shape[1]),
        },
    )

    # If no transform was given, add random transform and resolution
    if transform is None:
        xres = 1
        yres = 1
        transform = np.array([0, xres, 0, 0, 0, yres])

    # If input transform is affine, convert it to gdal
    if isinstance(transform, Affine):
        transform = np.array(transform.to_gdal())
    # Add transform to dataset
    trans_len = np.arange(0, len(transform))
    dataset.coords["trans_len"] = trans_len
    dataset["trans"] = xr.DataArray(data=transform, dims=["trans_len"])

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
    # Add image attributes to the image dataset
    dataset.attrs = {
        "no_data": no_data,
        "input_img": input_img,
        "crs": img_crs,
        "xres": transform[1],
        "yres": transform[5],
        "plani_unit": plani_unit,
        "zunit": new_zunit,
        "bounds": bounds,
        "geoid_path": geoid_path,
        "geoid_georef": geoid_georef,
        "source_rasterio": source_rasterio,
    }

    # If the georef is geoid, add geoid offset to the data
    if geoid_georef:
        # transform to ellipsoid
        geoid_offset = get_geoid_offset(dataset, geoid_path)
        dataset["im"].data += geoid_offset

    return dataset


def _crop_dataset_with_roi(
    src_static: rasterio.DatasetReader, roi: List[float]
) -> Tuple[np.ndarray, Affine]:
    """
    Transforms the input bounding box to polygon.
    Masks the input dataset and its transform.
    If the ROI is outside of the input dem, an exception is raised

    :param src_static: input source dataset in rasterio format
    :type src_static: rasterio.DatasetReader
    :param roi: region to crop
    :type roi: List[float]
    :return: masked dem and its transform
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
        new_cropped_static, new_cropped_static_transform = rasterio.mask.mask(
            src_static, [geom_like_polygon], all_touched=True, crop=True
        )
    except ValueError as roi_outside_dataset:
        logging.error(
            "Input ROI coordinates outside of the"
            " {} dataset scope.".format(src_static.files[0])
        )
        raise ValueError from roi_outside_dataset

    return new_cropped_static, new_cropped_static_transform


def save_dataset_to_tif(
    dataset: xr.Dataset, filename: str, new_array=None, no_data: float = -32768
) -> xr.Dataset:
    """
    Writes a Dataset in a tiff file.
    If new_array is set, new_array is used as data.

    :param dataset:  xarray.DataSet containing the variables :
            - im : 2D (row, col) xarray.DataArray float32
            - trans: 1D (trans_len) xarray.DataArray
    :type dataset: xr.Dataset
    :param filename:  output filename
    :type filename: str
    :param new_array:  new array to write
    :type new_array: np.ndarray or None
    :param no_data:  value of nodata to use
    :type no_data: float
    :return:  xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :rtype: xr.Dataset
    """

    # update from dataset
    previous_profile = {}
    previous_profile["crs"] = dataset.attrs["crs"]
    previous_profile["transform"] = Affine.from_gdal(
        dataset["trans"].data[0],
        dataset["trans"].data[1],
        dataset["trans"].data[2],
        dataset["trans"].data[3],
        dataset["trans"].data[4],
        dataset["trans"].data[5],
    )

    data = dataset["im"].data
    if new_array is not None:
        data = new_array

    if len(dataset["im"].shape) == 2:
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


def load_dem(
    path: str,
    nodata: float = None,
    geoid_georef: bool = False,
    geoid_path: Union[str, None] = None,
    zunit: str = "m",
    input_roi: Union[bool, dict, Tuple] = False,
) -> xr.Dataset:
    """
    Reads the input DEM path and parameters and generates
    the xr.Dataset to be handled in demcompare functions.

    :param path: path to dem
    :type path: str
    :param nodata: dem no data value
            (None by default and if set inside metadata)
    :type nodata: float or None
    :param geoid_georef: is dem's georef is geoid
    :type geoid_georef: bool
    :param geoid_path: optional path to local geoid, default is EGM96
    :type geoid_path: str or None
    :param zunit: dem z unit
    :type zunit: str
    :param input_roi: False if dem are to be fully loaded,
            other options are a dict roi
    :type input_roi: bool, dict or Tuple
    :return: dem  xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :rtype: xr.Dataset
    """

    # Read dem
    src_dem = rasterio.open(path)

    dem_trans = src_dem.transform
    bounds_dem = src_dem.bounds

    # TODO: clarify input_roi parameter
    # TODO: clarify geoid_georef and geoid_path parameters
    if input_roi is not False:
        # Use ROI
        if isinstance(input_roi, (tuple, list)):
            bounds_dem = input_roi

        elif isinstance(input_roi, dict):
            if (
                "left" in input_roi
                and "bottom" in input_roi
                and "right" in input_roi
                and "top" in input_roi
            ):
                # coordinates
                bounds_dem = (
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
                bounds_dem = rasterio.windows.bounds(window_dem, dem_trans)

            else:
                print("Not he right conventions for ROI")
    # create dataset
    dem = create_dataset(
        src_dem.read(1),
        dem_trans,
        src_dem.crs,
        path,
        bounds_dem,
        no_data=nodata,
        geoid_georef=geoid_georef,
        geoid_path=geoid_path,
        zunit=zunit,
        source_rasterio=src_dem,
    )

    return dem


def compute_altitude_diff(
    ref: xr.Dataset, dem_to_align: xr.Dataset
) -> xr.Dataset:
    """
    Compute altitude difference ref - dem_to_align and
    return it as an xr.Dataset with the dem_to_align
    georeferencing and attributes.

    :param ref: ref xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type ref: xr.Dataset
    :param dem_to_align: dem to alignxr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type dem_to_align: xr.Dataset
    :return: difference xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :rtype: xr.Dataset
    """
    diff = ref["im"].data - dem_to_align["im"].data
    diff_dataset = create_dataset(
        diff,
        transform=dem_to_align.trans.data,
        no_data=-32768,
        img_crs=dem_to_align.crs,
    )
    return diff_dataset


def reproject_dems(
    dem_to_align: xr.Dataset,
    ref: xr.Dataset,
    sampling_source: str = "dem_to_align",
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Reprojects both DEMs to common grid, common bounds and common georef origin.

    The common grid is defined by the sampling_source parameter, which defines
    which is the sampling of the output DEMs.

    The georef origin is always the ref's origin.

    If sampling_source is "ref":
        ref is cropped to the common bounds
        dem_to_align is cropped to the common bounds and resampled
    If sampling_source is "dem_to_align":
        ref is cropped to the common bounds and resampled
        dem_to_align is cropped to the common bounds

    :param dem_to_align: dem to alignxr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type dem_to_align: xr.Dataset
    :param ref: ref xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type ref: xr.Dataset
    :param sampling_source: 'ref' or 'dem_to_align', the sampling
        value of the output dems, by defaut "dem_to_align"
    :type sampling_source: str
    :return: reprojected and cropped ref and dem
                xr.DataSets containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :rtype: xr.Dataset, xr.Dataset
    """
    # The sampling value defines which DEM will undergo a change in resolution
    if sampling_source == "ref":
        interp = dem_to_align
        static = ref
    elif sampling_source == "dem_to_align":
        interp = ref
        static = dem_to_align

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
    new_cropped_static, new_cropped_static_transform = _crop_dataset_with_roi(
        src_static, intersection_roi
    )

    # Create cropped static dataset
    reproj_cropped_static = create_dataset(
        new_cropped_static,
        new_cropped_static_transform,
        img_crs=static_crs,
        no_data=static.attrs["no_data"],
        geoid_georef=static.attrs["geoid_georef"],
        geoid_path=static.attrs["geoid_path"],
        zunit=static.attrs["zunit"],
        input_img=static.attrs["input_img"],
        bounds=intersection_roi,
    )

    # Full_interp represent a dataset with the full interp image
    full_interp = create_dataset(
        interp["im"].data,
        interp.trans,
        img_crs=interp_crs,
        no_data=interp.attrs["no_data"],
        geoid_georef=interp.attrs["geoid_georef"],
        geoid_path=interp.attrs["geoid_path"],
        zunit=interp.attrs["zunit"],
        input_img=interp.attrs["input_img"],
        bounds=intersection_roi,
    )

    # Interp DEM is reprojected into the static DEM's georef-grid
    # Crop and resample are done in the interp DEM
    reproj_cropped_interp = reproject_dataset(
        full_interp, reproj_cropped_static, interp="bilinear"
    )
    # Update dataset input_img with interp old value
    reproj_cropped_interp.attrs["input_img"] = full_interp.attrs["input_img"]

    # Define reprojected ref and dem_to_align according to the sampling value
    if sampling_source == "ref":
        reproj_cropped_ref = reproj_cropped_static
        reproj_cropped_dem_to_align = reproj_cropped_interp
    elif sampling_source == "dem_to_align":
        reproj_cropped_ref = reproj_cropped_interp
        reproj_cropped_dem_to_align = reproj_cropped_static

    return reproj_cropped_dem_to_align, reproj_cropped_ref


def copy_dem(dem: xr.Dataset) -> xr.Dataset:
    """
    Returns a copy of the input dem.

    :param dem: input dem to copy, xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type dem: xr.Dataset
    :return dem_copy: copy of the input dem, xr.DataSet
                containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
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


def translate_to_coregistered_geometry(
    dem_to_align: xr.Dataset,
    ref: xr.Dataset,
    dx: int,
    dy: int,
    sampling_source: str = "dem_to_align",
    interpolator: str = "bilinear",
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Translate both DSMs to their coregistered geometry.

    Note that :

    The ref georef-origin is assumed to be the reference

    :param dem_to_align: dem_to_align
    :type dem_to_align: xr.Dataset
    :param ref: dataset, slave dem
    :type ref: xr.Dataset
    :param dx: f, dx value in pixels
    :type dx: int
    :param dy: f, dy value in pixels
    :type dy: int
    :param interpolator: gdal interpolator
    :type interpolator: str
    :return: coregistered DEM as datasets
    :rtype: xr.Dataset, xr.Dataset
    """

    # TODO: to be suppressed with the refacto coregistration
    # Translate the georef-origin of dem based on dx and dy values
    #   -> this makes dem coregistered on ref
    dem_to_align = translate_dataset(dem_to_align, dx, dy)
    # The sampling value defines which DEM will undergo a change in resolution
    if sampling_source == "ref":
        interp = dem_to_align
        static = ref
    elif sampling_source == "dem_to_align":
        interp = ref
        static = dem_to_align
    #
    # Intersect and reproject both dsms.
    #   -> intersect them to the biggest common grid
    #       now that they have been shifted
    #   -> static is then cropped with intersect so
    #       that it lies within intersect
    #       but is not resampled in the process
    #   -> reproject interp to static's georef-grid,
    #       the intersection grid sampled on static's grid
    #
    transform_static = Affine.from_gdal(
        static["trans"].data[0],
        static["trans"].data[1],
        static["trans"].data[2],
        static["trans"].data[3],
        static["trans"].data[4],
        static["trans"].data[5],
    )
    bounds_static = rasterio.transform.array_bounds(
        static["im"].data.shape[1], static["im"].data.shape[0], transform_static
    )

    transform_interp = Affine.from_gdal(
        interp["trans"].data[0],
        interp["trans"].data[1],
        interp["trans"].data[2],
        interp["trans"].data[3],
        interp["trans"].data[4],
        interp["trans"].data[5],
    )
    bounds_interp = rasterio.transform.array_bounds(
        interp["im"].data.shape[1], interp["im"].data.shape[0], transform_interp
    )

    intersection_roi = (
        max(bounds_static[0], bounds_interp[0]),
        max(bounds_static[1], bounds_interp[1]),
        min(bounds_static[2], bounds_interp[2]),
        min(bounds_static[3], bounds_interp[3]),
    )

    # crop static
    srs_static = rasterio.open(
        " ",
        mode="w+",
        driver="GTiff",
        width=static["im"].data.shape[1],
        height=static["im"].data.shape[0],
        count=1,
        dtype=static["im"].data.dtype,
        crs=static.attrs["georef"],
        transform=transform_static,
    )
    srs_static.write(static["im"].data, 1)
    new_cropped_static, new_cropped_static_transform = _crop_dataset_with_roi(
        srs_static, intersection_roi
    )

    # create datasets
    reproj_static = copy.copy(static)
    reproj_static["trans"].data = np.array(
        new_cropped_static_transform.to_gdal()
    )
    reproj_static["im"].data = new_cropped_static[0, :, :]

    # crop interp
    srs_interp = rasterio.open(
        " ",
        mode="w+",
        driver="GTiff",
        width=ref["im"].data.shape[1],
        height=ref["im"].data.shape[0],
        count=1,
        dtype=ref["im"].data.dtype,
        crs=ref.attrs["georef"],
        transform=transform_interp,
    )
    srs_interp.write(ref["im"].data, 1)
    new_cropped_interp, new_cropped_interp_transform = _crop_dataset_with_roi(
        srs_interp, intersection_roi
    )
    # create datasets
    reproj_interp = copy.copy(ref)
    reproj_interp["trans"].data = np.array(
        new_cropped_interp_transform.to_gdal()
    )
    reproj_interp["im"].data = new_cropped_interp[0, :, :]

    # Interp DEM is reprojected into the static DEM's georef-grid
    # Crop and resample are performed on the interp DEM
    reproj_interp = reproject_dataset(
        interp, reproj_interp, interp=interpolator
    )

    if sampling_source == "ref":
        reproj_ref = reproj_static
        reproj_dem_to_align = reproj_interp
    elif sampling_source == "dem_to_align":
        reproj_ref = reproj_interp
        reproj_dem_to_align = reproj_static
    return reproj_dem_to_align, reproj_ref
