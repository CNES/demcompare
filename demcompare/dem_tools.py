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
# pylint:disable=too-many-lines

# Standard imports
import copy
import logging
import os
import sys
from enum import Enum
from typing import Dict, Tuple, Union

import matplotlib.pyplot as mpl_pyplot

# Third party imports
import numpy as np
import rasterio
import xarray as xr
from astropy import units as u
from rasterio import Affine
from scipy.ndimage import convolve

from .dataset_tools import (
    compute_offset_adapting_factor,
    create_dataset,
    reproject_dataset,
)
from .img_tools import convert_pix_to_coord, crop_rasterio_source_with_roi
from .output_tree_design import get_out_file_path

DEFAULT_NODATA = -32768


def load_dem(
    path: str,
    nodata: float = None,
    band: int = 1,
    geoid_georef: bool = False,
    geoid_path: Union[str, None] = None,
    zunit: str = "m",
    input_roi: Union[bool, dict, Tuple] = False,
    classification_layers: Dict = None,
) -> xr.Dataset:
    """
    Reads the input DEM path and parameters and generates
    the DEM object as xr.Dataset to be handled in demcompare functions.

    A DEM can be any raster file opened by rasterio.

    :param path: path to dem (readable by rasterio)
    :type path: str
    :param nodata: forcing dem no data value
            (None by default and if set inside metadata)
    :type nodata: float or None
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
    :param classification_layers: input classification layers
    :type classification_layers: Dict or None
    :return: dem  xr.DataSet containing : (see dataset_tools for details)

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :rtype: xr.Dataset
    """

    # Open dem with rasterio
    # but source_rasterio is closed and tests bug
    src_dem = rasterio.open(path)
    source_rasterio = {}
    source_rasterio["source_dem"] = src_dem
    # Get rasterio transform :
    #  affine transformation matrix that maps pixel locations
    #    in (row, col) coordinates to (x, y) spatial positions
    dem_geotransform = src_dem.transform
    # Get rasterio BoundingBox(left, bottom, right, top)
    bounds_dem = src_dem.bounds

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
                left, bottom, right, top = rasterio.windows.bounds(
                    window_dem, dem_geotransform
                )
                bounds_dem = rasterio.coords.BoundingBox(
                    left, bottom, right, top
                )
            else:
                raise TypeError("Not the right conventions for ROI")
    # Get dem raster image from band image
    dem_image = src_dem.read(band)
    # Test nodata in DEM
    if np.all(dem_image == nodata):
        raise ValueError(
            f"All values in {source_rasterio['source_dem'].name} are NODATA"
        )

    classif_layers: Union[Dict, None] = None
    if classification_layers:
        classif_layers = {}
        classif_layers["names"] = []
        classif_layers["map_arrays"] = np.full(
            (
                dem_image.shape[0],
                dem_image.shape[1],
                len(classification_layers.keys()),
            ),
            np.nan,
            dtype=np.float32,
        )
        # Open the clasification layers with rasterio
        # and add the map_array to the layer dict
        for idx, [name, layer] in enumerate(classification_layers.items()):
            classif_rasterio_source = rasterio.open(layer["map_path"])
            map_array = classif_rasterio_source.read(band)
            if map_array.shape != dem_image.shape:
                logging.error(
                    "Input classification layer %s does not have the same size"
                    " as its reference dem.",
                    name,
                )
                sys.exit(1)
            classif_layers["map_arrays"][:, :, idx] = map_array
            classif_layers["names"].append(name)
            source_rasterio[name] = classif_rasterio_source

    # create dataset
    dem_dataset = create_dem(
        dem_image,
        dem_geotransform,
        src_dem.crs,
        path,
        bounds_dem,
        classification_layer_masks=classif_layers,
        nodata=nodata,
        geoid_georef=geoid_georef,
        geoid_path=geoid_path,
        zunit=zunit,
        source_rasterio=source_rasterio,
    )

    return dem_dataset


def copy_dem(dem: xr.Dataset) -> xr.Dataset:
    """
    Returns a copy of the input dem.

    :param dem: input dem to copy, xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type dem: xr.Dataset
    :return dem_copy: copy of the input dem, xr.DataSet
                containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
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
    dataset: xr.Dataset,
    filename: str,
    nodata: float = DEFAULT_NODATA,
) -> xr.Dataset:
    """
    Writes a Dataset in a tiff file.
    If new_array is set, new_array is used as data.
    Returns written dataset.

    :param dataset:  xarray.DataSet containing the variables :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type dataset: xr.Dataset
    :param filename:  output filename
    :type filename: str
    :param nodata:  value of nodata to use
    :type nodata: float
    :return:  xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
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
            source_ds.nodata = nodata
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
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type dataset: xr.Dataset
    :param x_offset: x offset
    :type x_offset: Union[float, int, ndarray]
    :param y_offset: y offset
    :type y_offset: Union[float, int, ndarray]
    :return: translated xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :rtype: xr.Dataset
    """
    dataset_translated = copy_dem(dataset)
    # Project the pixellic offset to input dataset's coordinates
    x_offset, y_offset = convert_pix_to_coord(
        dataset["georef_transform"].data, y_offset, x_offset
    )
    # To add an offset, the [0] and [3] positions
    # of the transform have to be modified
    dataset_translated["georef_transform"].data[0] = x_offset
    dataset_translated["georef_transform"].data[3] = y_offset

    if dataset.attrs["bounds"]:
        # Apply offset to bounds
        # Apply the bounds difference between its left/bottom
        # coordinate and the georef_transform origin
        # (which is zero if no ROI was specified)
        # to the new georef_transform origin
        bounds_x_offset = float(x_offset) + (
            dataset.attrs["bounds"].left - dataset["georef_transform"].data[0]
        )
        bounds_y_offset = float(y_offset) + (
            dataset.attrs["bounds"].top - dataset["georef_transform"].data[3]
        )

        dataset_translated.attrs["bounds"] = rasterio.coords.BoundingBox(
            float(bounds_x_offset),
            float(bounds_y_offset)
            - dataset.attrs["bounds"].top
            + dataset.attrs["bounds"].bottom,
            float(bounds_x_offset)
            + dataset.attrs["bounds"].right
            - dataset.attrs["bounds"].left,
            float(bounds_y_offset),
        )
    return dataset_translated


def create_dem(  # pylint: disable=too-many-arguments, too-many-branches
    data: np.ndarray,
    transform: Union[np.ndarray, rasterio.Affine] = None,
    img_crs: Union[rasterio.crs.CRS, None] = None,
    input_img: Union[str, None] = None,
    bounds: rasterio.coords.BoundingBox = None,
    classification_layer_masks: Union[Dict, xr.DataArray] = None,
    nodata: float = None,
    geoid_georef: bool = False,
    geoid_path: Union[str, None] = None,
    zunit: str = "m",
    source_rasterio: Dict[str, rasterio.DatasetReader] = None,
) -> xr.Dataset:
    """
    Creates dem from input array and transform.

    The demcompare DEM is an xarray Dataset containing:
                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray

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
    :param classification_layer_masks: classification layers
    :type classification_layer_masks: Dict,  xr.DataArray or None
    :param nodata: nodata value in the image
    :type nodata: float or None
    :param geoid_georef: if dem's georef is geoid
    :type geoid_georef: bool
    :param geoid_path: optional path to local geoid, default is EGM96
    :type geoid_path: str or None
    :param zunit: unit
    :type zunit: str
    :param source_rasterio: rasterio dataset reader object
    :type source_rasterio: Dict[str,rasterio.DatasetReader] or None
    :return: xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :rtype: xr.Dataset
    """

    # If source_rasterio (DatasetReader is given), set img_ds variable
    img_ds = None
    if source_rasterio:
        img_ds = source_rasterio["source_dem"]

    # If no nodata value was given
    # If img_ds exists and has nodatavals,
    # Otherwise the
    # DEFAULT_NODATA value is used
    if nodata is None:
        nodata = DEFAULT_NODATA
        if img_ds:
            meta_nodata = img_ds.nodatavals[0]
            if meta_nodata is not None:
                nodata = meta_nodata

    # If input data has three dimensions, flatten
    if len(data.shape) == 3:
        # to dim 2
        dim_single = np.where(data.shape) == 1
        if dim_single == 0:
            data = data[0, :, :]
        if dim_single == 2:
            data = data[:, :, 0]

    # Convert nodata values to nan
    data = data.astype(np.float32)
    data[data == nodata] = np.nan

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
        nodata,
        geoid_path,
        plani_unit,
        new_zunit,
        source_rasterio,
        classification_layer_masks,
    )

    return dataset


class SamplingSourceParameter(Enum):
    """
    Enum type definition for sampling_source parameter
    value are sec or ref to choose in reproject_dems
    """

    SEC = "sec"
    REF = "ref"


def reproject_dems(
    sec: xr.Dataset,
    ref: xr.Dataset,
    initial_shift_x: Union[int, float] = 0,
    initial_shift_y: Union[int, float] = 0,
    sampling_source: str = SamplingSourceParameter.SEC.value,
) -> Tuple[xr.Dataset, xr.Dataset, Tuple[float, float]]:
    """
    Reprojects both DEMs to common grid, common bounds and common georef origin.

    The common grid, bounds, georef origin are defined
    by the sampling_source parameter.
    It defines which is the sampling of the output DEMs.

    - If sampling_source is "ref":
      ref is cropped to the common bounds
      sec is cropped to the common bounds and resampled
    - If sampling_source is "sec":
      ref is cropped to the common bounds and resampled
      sec is cropped to the common bounds

    :param sec: dem to align xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type sec: xr.Dataset
    :param ref: ref xr.DataSet containing :

                - im : 2D (row, col) xarray.DataArray float32
                - trans: 1D (trans_len) xarray.DataArray
    :type ref: xr.Dataset
    :param initial_shift_x: optional initial shift x
    :type initial_shift_x: Union[int, float]
    :param initial_shift_y: optional initial shift y
    :type initial_shift_y: Union[int, float]
    :param sampling_source: 'ref' or 'sec', the sampling
                 value of the output dems, by defaut "sec"
    :type sampling_source: str
    :return: reproj_cropped_sec xr.DataSet,
                 reproj_cropped_ref xr.DataSet, adapting_factor.
                 The xr.Datasets containing :

                 - im : 2D (row, col) xarray.DataArray float32
                 - trans: 1D (trans_len) xarray.DataArray
    :rtype: xr.Dataset, xr.Dataset, Tuple[float, float]
    """
    if sampling_source == SamplingSourceParameter.REF.value:
        interp = sec
        static = ref
        # Compute offset adapting factor for later adapting the
        # offset to the original dem resolution
        adapting_factor = compute_offset_adapting_factor(sec, ref)
    else:  # sampling_source == SamplingSourceParameter.SEC.value:
        interp = ref
        static = sec
        # If sampling value is sec, adapting
        # factor is 1 (no adaptation needed)
        adapting_factor = (1.0, 1.0)

    # Reproject if input image is inversed top bottom or left right
    # to have the same consistent (left, bottom, right, top) reference
    # than interp projected bounds (orientation bug otherwise)
    transformed_static_bounds = rasterio.warp.transform_bounds(
        static.attrs["crs"],
        static.attrs["crs"],
        static.attrs["bounds"][0],
        static.attrs["bounds"][1],
        static.attrs["bounds"][2],
        static.attrs["bounds"][3],
    )

    # Project bounds to static_crs
    transformed_interp_bounds = rasterio.warp.transform_bounds(
        interp.attrs["crs"],
        static.attrs["crs"],
        interp.attrs["bounds"][0],
        interp.attrs["bounds"][1],
        interp.attrs["bounds"][2],
        interp.attrs["bounds"][3],
    )

    # Obtain intersection roi
    if rasterio.coords.disjoint_bounds(
        transformed_static_bounds, transformed_interp_bounds
    ):
        raise NameError("ERROR: ROIs do not intersect")
    intersection_roi = rasterio.coords.BoundingBox(
        max(transformed_static_bounds[0], transformed_interp_bounds[0]),
        max(transformed_static_bounds[1], transformed_interp_bounds[1]),
        min(transformed_static_bounds[2], transformed_interp_bounds[2]),
        min(transformed_static_bounds[3], transformed_interp_bounds[3]),
    )

    # Crop static dem
    # rasterio.mask.mask function needs the src read by rasterio
    src_static = static.attrs["source_rasterio"]["source_dem"]
    (
        new_cropped_static,
        new_cropped_static_transform,
    ) = crop_rasterio_source_with_roi(src_static, intersection_roi)

    # Crop static classification layers
    # rasterio.mask.mask function needs the src read by rasterio
    if "indicator" in static.coords:
        cropped_static_classif = np.full(
            (
                new_cropped_static.shape[1],
                new_cropped_static.shape[2],
                len(static["classification_layer_masks"].coords["indicator"]),
            ),
            np.nan,
            dtype=np.float32,
        )
        for idx, indicator in enumerate(
            static["classification_layer_masks"].coords["indicator"].data
        ):
            src_classif = static.attrs["source_rasterio"][indicator]
            (
                new_cropped_classif,
                _,
            ) = crop_rasterio_source_with_roi(src_classif, intersection_roi)
            cropped_static_classif[:, :, idx] = new_cropped_classif
        # Remove confidence_measure dataArray from the dataset to update it
        indicator = list(static.coords["indicator"].data)
        static = static.drop_dims("indicator")
        coords_classification_layers = {
            "row": np.arange(new_cropped_static.shape[1]),
            "col": np.arange(new_cropped_static.shape[2]),
            "indicator": indicator,
        }

        static["classification_layer_masks"] = xr.DataArray(
            data=cropped_static_classif,
            coords=coords_classification_layers,
            dims=["row", "col", "indicator"],
        )

    # Create cropped static dem
    reproj_cropped_static = create_dem(
        new_cropped_static,
        new_cropped_static_transform,
        img_crs=static.attrs["crs"],
        nodata=static.attrs["nodata"],
        geoid_path=static.attrs["geoid_path"],
        zunit=static.attrs["zunit"],
        input_img=static.attrs["input_img"],
        bounds=intersection_roi,
        classification_layer_masks=static.classification_layer_masks
        if "indicator" in static.coords
        else None,
    )

    # Full_interp represent a dem with the full interp image
    full_interp = create_dem(
        interp["image"].data,
        interp.georef_transform,
        img_crs=interp.attrs["crs"],
        nodata=interp.attrs["nodata"],
        geoid_path=interp.attrs["geoid_path"],
        zunit=interp.attrs["zunit"],
        input_img=interp.attrs["input_img"],
        bounds=intersection_roi,
        classification_layer_masks=interp.classification_layer_masks
        if "indicator" in interp.coords
        else None,
    )

    # Translate sec according to the initial shift
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

    # Define reprojected ref and sec according to the sampling value
    if sampling_source == SamplingSourceParameter.REF.value:
        reproj_cropped_ref = reproj_cropped_static
        reproj_cropped_sec = reproj_cropped_interp
    else:  # sampling_source == SamplingSourceParameter.SEC.value:
        reproj_cropped_ref = reproj_cropped_interp
        reproj_cropped_sec = reproj_cropped_static

    return reproj_cropped_sec, reproj_cropped_ref, adapting_factor


def compute_dems_diff(dem_1: xr.Dataset, dem_2: xr.Dataset) -> xr.Dataset:
    """
    Compute altitude difference dem_1 - dem_2 and
    return it as an xr.Dataset with the dem_2
    georeferencing and attributes.

    :param dem_1: dem_1 xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type dem_1: xr.Dataset
    :param dem_2: dem_2 xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :type dem_2: xr.Dataset
    :return: difference xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :rtype: xr.Dataset
    """
    diff_raster = dem_1["image"].data - dem_2["image"].data

    diff_dem = create_dem(
        diff_raster,
        transform=dem_2.georef_transform.data,
        nodata=dem_1.attrs["nodata"],
        img_crs=dem_2.crs,
        bounds=dem_2.bounds,
    )
    return diff_dem


def compute_waveform(
    dem: xr.Dataset, output_dir: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Metric computation method

    :param dem: input data to compute the metric
    :type dem: xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :param output_dir: optional output directory
    :type output_dir: str
    :return: the computed row and col waveforms
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    data = dem["image"].data
    mean_row = np.nanmean(data, axis=0)
    row_waveform = data - mean_row
    # for axis == 1, we need to transpose the array to substitute it
    # to final_dh.r otherwise 1D array stays row array
    mean_col = np.nanmean(data, axis=1)

    mean_col = np.transpose(
        np.ones((1, mean_col.size), dtype=np.float32) * mean_col
    )
    col_waveform = data - mean_col

    if output_dir:
        # Save waveform dems
        save_dem(
            create_dem(
                row_waveform,
                transform=dem.georef_transform.data,
                img_crs=dem.crs,
                nodata=DEFAULT_NODATA,
            ),
            os.path.join(
                output_dir,
                get_out_file_path("dh_row_wise_wave_detection.tif"),
            ),
        )
        save_dem(
            create_dem(
                col_waveform,
                transform=dem.georef_transform.data,
                img_crs=dem.crs,
                nodata=DEFAULT_NODATA,
            ),
            os.path.join(
                output_dir,
                get_out_file_path("dh_col_wise_wave_detection.tif"),
            ),
        )

    return row_waveform, col_waveform


def compute_alti_diff_for_stats(  # pylint:disable=too-many-branches
    ref: xr.Dataset, sec: xr.Dataset
) -> xr.Dataset:
    """
    Computes the difference dem between the two inputs (ref - sec)
    and accumulates the classification layers of each dem

    :param ref: ref dem
    :type ref: xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :param sec: sec dem
    :type ref: xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray

    :return: the difference dem
    :rtype: xr.Dataset containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    """
    # Compute altitude diff dataset
    altitude_diff = compute_dems_diff(ref, sec)
    support_list = []
    # Initialize classif_layers_datarray
    classif_layers_datarray = None
    if "classification_layer_masks" in ref:
        # If classification layers in ref,
        # Add the layer on the dataarray
        classif_layers_datarray = copy.deepcopy(
            ref["classification_layer_masks"]
        )
        for _ in classif_layers_datarray.coords["indicator"].data:
            support_list.append("ref")
    # Add the fusion information on the altitude diff dataset
    if "fusion_layers" in ref.attrs:
        altitude_diff.attrs["fusion_layers"] = ref.attrs["fusion_layers"]
    # If slope is present, add the dataarray
    if "ref_slope" in ref:
        ref_slope = copy.deepcopy(ref["ref_slope"])
        altitude_diff["ref_slope"] = ref_slope

    if "classification_layer_masks" in sec:
        # If classification layers in sec,
        # Add the layer on the dataarray
        if isinstance(classif_layers_datarray, xr.DataArray):
            # If the dataarray already existed,
            # update it
            nb_row = classif_layers_datarray.shape[0]
            nb_col = classif_layers_datarray.shape[1]
            nb_indicator = (
                classif_layers_datarray.shape[2]
                + sec["classification_layer_masks"].shape[2]
            )

            # Add a new indicator to the DataArray
            updated_data = np.full(
                (nb_row, nb_col, nb_indicator), np.nan, dtype=np.float32
            )
            # Ref classification layers
            updated_data[
                :, :, : -sec["classification_layer_masks"].shape[2]
            ] = classif_layers_datarray.data
            # Sec classification layers
            updated_data[:, :, classif_layers_datarray.shape[2] :] = sec[
                "classification_layer_masks"
            ].data

            indicator = np.copy(classif_layers_datarray.coords["indicator"])
            for new_indicator in sec["classification_layer_masks"].coords[
                "indicator"
            ]:
                indicator = np.append(indicator, new_indicator)

            expanded_coords = [
                classif_layers_datarray.coords["row"],
                classif_layers_datarray.coords["col"],
                indicator,
            ]

            classif_layers_datarray = xr.DataArray(
                data=updated_data,
                coords=expanded_coords,
                dims=["row", "col", "indicator"],
            )

        else:
            classif_layers_datarray = copy.deepcopy(
                sec["classification_layer_masks"]
            )
        for _ in sec["classification_layer_masks"].coords["indicator"].data:
            support_list.append("sec")
    # If slope is present, add the dataarray
    # When computing the slope for a single dem, the indicator
    # is always ref_slope by default, so we adapt it to sec_slope
    if "ref_slope" in sec:
        sec_slope = copy.deepcopy(sec["ref_slope"])
        altitude_diff["sec_slope"] = sec_slope
    # Add the fusion information on the altitude diff dataset
    if "fusion_layers" in sec.attrs:
        if "fusion_layers" in altitude_diff:
            altitude_diff.attrs["fusion_layers"].append(
                sec.attrs["fusion_layers"]
            )
        else:
            altitude_diff.attrs["fusion_layers"] = sec.attrs["fusion_layers"]
    # Add the dataarray on the altitude diff dataset
    altitude_diff["classification_layer_masks"] = classif_layers_datarray
    # Add the support_list as an attribute
    altitude_diff.attrs["support_list"] = support_list

    return altitude_diff


def compute_dem_slope(dataset: xr.Dataset, degree: bool = False) -> xr.Dataset:
    """
    Computes DEM's slope
    Slope is presented here :
    http://pro.arcgis.com/ \
            fr/pro-app/tool-reference/spatial-analyst/how-aspect-works.htm

    :param dataset: dataset
    :type dataset: xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
    :param degree:  True if is in degree
    :type degree: bool
    :return: slope
    :rtype: np.ndarray
    """

    def _get_orthodromic_distance(
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
        # we need to compute resolution between each point

        # Create grid of image's size
        ny, nx = dataset["image"].data.shape
        xp = np.arange(nx)
        yp = np.arange(ny)
        xp, yp = np.meshgrid(xp, yp)
        # Convert all pixels on grid to lat lon
        lon, lat = convert_pix_to_coord(
            dataset["georef_transform"].data, yp, xp
        )
        lonr = np.roll(lon, 1, 1)
        latl = np.roll(lat, 1, 0)
        # Get distance between all pixels
        distx = _get_orthodromic_distance(lon, lat, lonr, lat)
        disty = _get_orthodromic_distance(lon, lat, lon, latl)

        # deal with singularities at edges
        distx[:, 0] = distx[:, 1]
        disty[0] = disty[1]
    else:
        # if resolution is define, all pixels consider this distance
        distx = np.abs(dataset.attrs["xres"])
        disty = np.abs(dataset.attrs["yres"])

    # Convolution kernel
    conv_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    conv_y = conv_x.transpose()

    # Now we do the convolutions :
    gx = convolve(dataset["image"].data, conv_x, mode="reflect")
    gy = convolve(dataset["image"].data, conv_y, mode="reflect")

    # And eventually we do compute tan(slope) and aspect
    tan_slope = np.sqrt((gx / distx) ** 2 + (gy / disty) ** 2) / 8
    slope = np.arctan(tan_slope)

    # Just simple unit change as required
    if degree is False:
        slope *= 100
    else:
        slope = (slope * 180) / np.pi

    # Add slope as a DataArray
    # Slope
    coords_slope = [dataset.coords["row"], dataset.coords["col"]]
    # Add computed slope in the 3D classification layer format
    data = np.full((slope.shape[0], slope.shape[1]), np.nan, dtype=np.float32)
    data[:, :] = slope
    # Create the dataarray
    # In case there is a single dem, we name the datarray
    # as ref_slope. If there are two arrays, the sec slope
    # will be renamed by the compute_alti_diff_for_stats function
    # to sec_slope
    dataset["ref_slope"] = xr.DataArray(
        data=data,
        coords=coords_slope,
        dims=["row", "col"],
    )

    return dataset


def compute_and_save_image_plots(
    dem: xr.Dataset, plot_path: str, title: str = None, dem_path: str = None
):
    """
    Compute and save dem plot and optionally the original dem image tif.
    Saves dem tif if the dem_path parameter is given.

    :param dem: name to save the plots
    :type dem: str
    :param plot_path: path to save the plots
    :type plot_path: str
    :param title: optional plot title
    :type title: str
    :param dem_path: optional dem path
      to save the original tif file
    :type dem_path: str
    :returns: None
    """

    # Save image
    if dem_path:
        save_dem(
            dem,
            dem_path,
        )

    # Create and save plot using the dem_plot function

    # Compute mean
    mu = np.nanmean(dem["image"].data)
    # Compute std
    sigma = np.std(dem["image"].data)

    # Plot
    fig, fig_ax = mpl_pyplot.subplots(figsize=(7.0, 8.0))
    if title:
        fig_ax.set_title(title, fontsize="large")
    im1 = fig_ax.imshow(
        dem["image"].data,
        cmap="terrain",
        vmin=mu - sigma,
        vmax=mu + sigma,
    )
    fig.colorbar(im1, label="Elevation differences (m)")
    fig.text(
        0.15,
        0.15,
        f"Image diff view: [Min, Max]=[{mu - sigma}.2f, {mu + sigma}.2f]",
        fontsize="medium",
    )
    # Save plot
    mpl_pyplot.savefig(plot_path, dpi=100, bbox_inches="tight")
    mpl_pyplot.close()


def verify_fusion_layers(dem: xr.Dataset, classif_cfg: Dict, support: str):
    """
    Verifies that the input configuration and input dem contain the
    input necessary layers for the fusion classification.

    :param dem: name to save the plots
    :type dem: str
    :param classif_cfg: classification layers configuration
    :type classif_cfg: Dict
    :param support: fusion support, ref or sec
    :type support: str
    :returns: None
    """
    classif_names = list(classif_cfg.keys())
    fusion_layers = []
    for _, clayer in classif_cfg.items():
        if clayer["type"] == "fusion":
            if support in clayer:
                fusion_layers.append(clayer)
    for fusion_layer in fusion_layers:
        for layer_to_fusion in fusion_layer[support]:
            if layer_to_fusion not in classif_names:
                logging.error(
                    "Input layer to be fused %s not defined"
                    " in the classification layers configuration.",
                    layer_to_fusion,
                )
                raise ValueError
            layer_type = classif_cfg[layer_to_fusion]["type"]
            # If the layer type is slope, verify that the slope
            # has been computed on the dem
            if layer_type == "slope":
                if "ref_slope" not in dem:
                    logging.error(
                        "Input layer to be fused is type slope, "
                        " but slope has not been computed on the input dem."
                    )
                    raise ValueError
            # If the layer type is segmentation, verify that the layer map
            # has been loaded on the dem
            else:
                if "classification_layer_masks" in dem:
                    dem_classif_names = list(
                        dem["classification_layer_masks"]
                        .coords["indicator"]
                        .data
                    )
                    if layer_to_fusion not in dem_classif_names:
                        logging.error(
                            "Input layer to be fused %s, "
                            " is not defined on the input dem.",
                            layer_to_fusion,
                        )
                        raise ValueError
                else:
                    logging.error(
                        "Input layer to be fused %s, "
                        " is not defined on the input dem.",
                        layer_to_fusion,
                    )
                    raise ValueError
