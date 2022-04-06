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
This module contains functions associated to raster images.
"""

# Standard imports
import copy
import logging
import os
from typing import Dict, List, Tuple, Union

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
from scipy.ndimage import filters


def read_image(path: str, band: int = 1) -> np.ndarray:
    """
    Read image as array

    :param path: path
    :type path: str
    :param band: numero of band to extract
    :type band: int
    :return: band array
    :rtype: np.ndarray
    """
    img_ds = rasterio.open(path)
    data = img_ds.read(band)
    return data


def pix_to_coord(
    transform_array: Union[List, np.ndarray],
    row: Union[int, np.ndarray],
    col: Union[List, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform pixels to coordinates

    :param transform_array: Transform
    :type transform_array: List or np.ndarray
    :param row: row
    :type row: int or np.ndarray
    :param col: column
    :type col: List or np.ndarray
    :return: x,y
    :rtype: np.ndarray, np.ndarray
    """
    transform = Affine.from_gdal(
        transform_array[0],
        transform_array[1],
        transform_array[2],
        transform_array[3],
        transform_array[4],
        transform_array[5],
    )
    # Set the offset to ul (upper left)
    x, y = rasterio.transform.xy(transform, row, col, offset="ul")

    if not isinstance(x, int):
        x = np.array(x)
        y = np.array(y)

    return x, y


def reproject_dataset(
    dataset: xr.Dataset, from_dataset: xr.Dataset, interp: str = "bilinear"
) -> xr.Dataset:
    """
    Reproject dataset, and return the corresponding xarray.DataSet

    :param dataset: Dataset to reproject
    :type dataset: xr.Dataset
    :param from_dataset: Dataset to get projection from
    :type from_dataset: xr.Dataset
    :param interp: interpolation method
    :type interp: str
    :return: reprojected dataset
    :rtype: xr.Dataset
    """

    # Copy dataset
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

    source_array = dataset["im"].data
    dest_array = np.zeros_like(from_dataset["im"].data)
    dest_array[:, :] = -9999

    src_crs = rasterio.crs.CRS.from_dict(dataset.attrs["georef"])
    dst_crs = rasterio.crs.CRS.from_dict(from_dataset.attrs["georef"])

    # reproject
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

    # change output dataset
    dest_array[dest_array == -9999] = np.nan
    reprojected_dataset["im"].data = dest_array
    reprojected_dataset.attrs["no_data"] = dataset.attrs["no_data"]

    return reprojected_dataset


def read_img(
    img: str,
    no_data: float = None,
    georef_grid: str = "WGS84",
    geoid_path: Union[str, None] = None,
    zunit: str = "m",
    load_data: bool = False,
) -> xr.Dataset:
    """
    Read image and transform and return the corresponding xarray.DataSet

    :param img: Path to the image
    :type img: str
    :param no_data: no_data value in the image
    :type no_data: float
    :param ref_georef_grid: WGS84 or geoid
    :type ref_georef_grid: str
    :param geoid_path: optional, path to local geoid
    :type ref_georef_grid: str or None
    :param zunit: unit
    :type zunit: str
    :param load_data: load as dem
    :type load_data: bool
    :return: dataset containing the variables :
            - im : 2D (row, col) xarray.DataArray float32
            - trans 1D xarray.DataArray float32
    :rtype: xr.Dataset
    """
    img_ds = rasterio.open(img)
    data = img_ds.read(1)
    transform = img_ds.transform

    dataset = create_dataset(
        data,
        transform,
        img,
        no_data=no_data,
        georef_grid=georef_grid,
        geoid_path=geoid_path,
        zunit=zunit,
        load_data=load_data,
    )

    return dataset


def create_dataset(
    data: np.ndarray,
    transform: np.ndarray,
    img: str,
    no_data: float = None,
    georef_grid: str = "WGS84",
    geoid_path: Union[str, None] = None,
    zunit: str = "m",
    load_data: bool = False,
) -> xr.Dataset:
    """
    Create dataset from array and transform,
    and return the corresponding xarray.DataSet

    :param data: image data
    :type data: np.ndarray
    :param transform: image data
    :type transform: np.ndarray
    :param img: image path
    :type img: str
    :param no_data: no_data value in the image
    :type no_data: float or None
    :param georef_grid: WGS84 or geoid
    :type georef_grid: str
    :param geoid_path: optional path to local geoid, default is EGM96
    :type geoid_path: str or None
    :param zunit: unit
    :type zunit: str
    :param load_data: load as dem
    :type load_data: bool
    :return: xarray.DataSet containing the variables :
            - im : 2D (row, col) xarray.DataArray float32
    :type data: xr.Dataset
    """

    img_ds = rasterio.open(img)

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

    dataset = xr.Dataset(
        {"im": (["row", "col"], data.astype(np.float32))},
        coords={
            "row": np.arange(data.shape[0]),
            "col": np.arange(data.shape[1]),
        },
    )

    transform = np.array(transform.to_gdal())
    # Add transform
    trans_len = np.arange(0, len(transform))
    dataset.coords["trans_len"] = trans_len
    dataset["trans"] = xr.DataArray(data=transform, dims=["trans_len"])

    # The image's crs is used as georef. The georef_grid
    # argument is used to add an offset if necessary
    img_crs = img_ds.crs
    # get plani unit
    if img_crs.is_geographic:
        plani_unit = u.deg
    else:
        plani_unit = u.m

    # Add image conf to the image dataset
    # Add resolution, and units
    dataset.attrs = {
        "no_data": no_data,
        "input_img": img,
        "georef": img_crs,
        "xres": transform[1],
        "yres": transform[5],
        "plani_unit": plani_unit,
        "zunit": new_zunit,
    }

    if load_data is not False:
        # If the georef is geoid, add geoid offset
        if georef_grid == "geoid":
            # transform to ellipsoid
            geoid_offset = get_geoid_offset(dataset, geoid_path)
            dataset["im"].data += geoid_offset

    return dataset


def read_img_from_array(
    img_array: np.ndarray,
    from_dataset: xr.Dataset = None,
    no_data: float = None,
) -> xr.Dataset:
    """
    Read image, and return the corresponding xarray.DataSet.
    If from_dataset is None defaults attributes are set.

    :param img_array: array
    :type img_array: np.ndarray
    :param from_dataset: dataset to copy
    :type from_dataset: xr.Dataset or None
    :param no_data: no_data value in the image
    :type no_data: float or None
    :return: xarray.DataSet containing the variables :
            - im : 2D (row, col) xarray.DataArray float32
    :rtype: xr.Dataset
    """

    data = np.copy(img_array)

    # Manage nodata
    if no_data is None:
        no_data = -9999

    data[data == no_data] = np.nan

    if from_dataset is None:
        dataset = xr.Dataset(
            {"im": (["row", "col"], data.astype(np.float32))},
            coords={
                "row": np.arange(data.shape[0]),
                "col": np.arange(data.shape[1]),
            },
        )

        # add random resolution
        dataset.attrs["xres"] = 1
        dataset.attrs["yres"] = 1

        # add nodata
        dataset.attrs["no_data"] = no_data
    else:
        dataset = copy.deepcopy(from_dataset)
        dataset["im"] = None
        dataset.coords["row"] = np.arange(data.shape[0])
        dataset.coords["col"] = np.arange(data.shape[1])
        dataset["im"] = xr.DataArray(
            data=data.astype(np.float32), dims=["row", "col"]
        )

    return dataset


def load_dems(
    ref_path: str,
    dem_path: str,
    ref_nodata: float = None,
    dem_nodata: float = None,
    ref_georef_grid: str = "WGS84",
    dem_georef_grid: str = "WGS84",
    ref_geoid_path: Union[str, None] = None,
    dem_geoid_path: Union[str, None] = None,
    ref_zunit: str = "m",
    dem_zunit: str = "m",
    load_data: Union[bool, dict, Tuple] = True,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Loads both DEMs

    :param ref_path: path to ref dem
    :type ref_path: str
    :param dem_path: path to sec dem
    :type dem_path: str
    :param ref_nodata: ref no data value
            (None by default and if set inside metadata)
    :type ref_nodata: float or None
    :param dem_nodata: dem no data value
            (None by default and if set inside metadata)
    :type dem_nodata: float or None
    :param ref_georef_grid: ref georef (either WGS84 -default- or geoid)
    :type ref_georef_grid: str
    :param dem_georef_grid: dem georef (either WGS84 -default- or geoid)
    :type dem_georef_grid: str
    :param ref_geoid_path: optional path to local geoid, default is EGM96
    :type ref_geoid_path: str or None
    :param dem_geoid_path: optional path to local geoid, default is EGM96
    :type dem_geoid_path: str or None
    :param ref_zunit: ref z unit
    :type ref_zunit: str
    :param dem_zunit: dem z unit
    :type dem_zunit: str
    :param load_data: True if dem are to be fully loaded,
            other options are False or a dict roi
    :type load_data: bool, dict or Tuple
    :return: ref and dem datasets
    :rtype: xr.Dataset, xr.Dataset
    """

    # Get roi of dem
    src_dem = rasterio.open(dem_path)
    dem_crs = src_dem.crs

    dem_trans = src_dem.transform
    bounds_dem = src_dem.bounds

    if load_data is not True:
        # Use ROI
        if isinstance(load_data, (tuple, list)):
            bounds_dem = load_data

        elif isinstance(load_data, dict):
            if (
                "left" in load_data
                and "bottom" in load_data
                and "right" in load_data
                and "top" in load_data
            ):
                # coordinates
                bounds_dem = (
                    load_data["left"],
                    load_data["bottom"],
                    load_data["right"],
                    load_data["top"],
                )
            elif (
                "x" in load_data
                and "y" in load_data
                and "w" in load_data
                and "h" in load_data
            ):
                # coordinates
                window_dem = rasterio.windows.Window(
                    load_data["x"],
                    load_data["y"],
                    load_data["w"],
                    load_data["h"],
                )
                bounds_dem = rasterio.windows.bounds(window_dem, dem_trans)

            else:
                print("Not he right conventions for ROI")

    # Reproject if input image is inversed top bottom or left right
    # to have the same consistent (left, bottom, right, top) reference
    # than ref projected bounds (orientation bug otherwise)
    transformed_dem_bounds = rasterio.warp.transform_bounds(
        dem_crs,
        dem_crs,
        bounds_dem[0],
        bounds_dem[1],
        bounds_dem[2],
        bounds_dem[3],
    )

    # Get roi of ref

    src_ref = rasterio.open(ref_path)
    ref_crs = src_ref.crs

    bounds_ref = src_ref.bounds

    # Project bounds to dem_crs reference
    transformed_ref_bounds = rasterio.warp.transform_bounds(
        ref_crs,
        dem_crs,
        bounds_ref[0],
        bounds_ref[1],
        bounds_ref[2],
        bounds_ref[3],
    )

    # intersect roi
    if rasterio.coords.disjoint_bounds(
        transformed_dem_bounds, transformed_ref_bounds
    ):
        raise NameError("ERROR: ROIs do not intersect")
    intersection_roi = (
        max(transformed_dem_bounds[0], transformed_ref_bounds[0]),
        max(transformed_dem_bounds[1], transformed_ref_bounds[1]),
        min(transformed_dem_bounds[2], transformed_ref_bounds[2]),
        min(transformed_dem_bounds[3], transformed_ref_bounds[3]),
    )

    # get  crop
    polygon_roi = bounding_box_to_polygon(
        intersection_roi[0],
        intersection_roi[1],
        intersection_roi[2],
        intersection_roi[3],
    )
    geom_like_polygon = {"type": "Polygon", "coordinates": [polygon_roi]}

    # crop dem
    new_cropped_dem, new_cropped_dem_transform = rasterio.mask.mask(
        src_dem, [geom_like_polygon], all_touched=True, crop=True
    )

    # create datasets

    dem = create_dataset(
        new_cropped_dem,
        new_cropped_dem_transform,
        dem_path,
        no_data=dem_nodata,
        georef_grid=dem_georef_grid,
        geoid_path=dem_geoid_path,
        zunit=dem_zunit,
        load_data=load_data,
    )

    # full_ref represent a dataset with the full image
    full_ref = create_dataset(
        src_ref.read(1),
        src_ref.transform,
        ref_path,
        no_data=ref_nodata,
        georef_grid=ref_georef_grid,
        geoid_path=ref_geoid_path,
        zunit=ref_zunit,
        load_data=load_data,
    )

    # Reference DEM is reprojected into the sec DEM's georef-grid
    # Crop and resample are done in the reference DEM
    ref = reproject_dataset(full_ref, dem, interp="bilinear")
    # update dataset input_img with ref old value
    ref.attrs["input_img"] = full_ref.attrs["input_img"]

    return ref, dem


def bounding_box_to_polygon(
    left: float, bottom: float, right: float, top: float
) -> List[List[float]]:
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
    :rtype: List[List[float]]
    """

    polygon = [
        [left, bottom],
        [right, bottom],
        [right, top],
        [left, top],
        [left, bottom],
    ]

    return polygon


def translate(
    dataset: xr.Dataset, x_offset: float, y_offset: float
) -> xr.Dataset:
    """
    Modify transform from dataset

    :param dataset:
    :type dataset: xr.Dataset
    :param x_offset: x offset
    :type x_offset: float
    :param y_offset: y offset
    :type y_offset: float
    :return: translated dataset
    :rtype: xr.Dataset
    """
    dataset_translated = copy.copy(dataset)

    x_off, y_off = pix_to_coord(dataset["trans"].data, y_offset, x_offset)
    dataset_translated["trans"].data[0] = x_off
    dataset_translated["trans"].data[3] = y_off

    return dataset_translated


def translate_to_coregistered_geometry(
    dem: xr.Dataset,
    ref: xr.Dataset,
    dx: int,
    dy: int,
    interpolator: str = "bilinear",
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Translate both DSMs to their coregistered geometry.

    Note that :

    The ref georef-origin is assumed to be the reference

    The ref shall be the one resampled at dem's
    georef-grid as it supposedly is the cleaner one.

    Hence, dem is only cropped, and ref is
    projected on dem's georef-grid, so it might be resampled.
    However, the dem's georef-origin is translated
    to the ref's georef-origin, which is considered the reference.

    :param dem: dataset, master dem
    :type dem: xr.Dataset
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

    #
    # Translate the georef-origin of dem based on dx and dy values
    #   -> this makes dem coregistered on ref
    dem = translate(dem, dx, dy)

    #
    # Intersect and reproject both dsms.
    #   -> intersect them to the biggest common grid
    #       now that they have been shifted
    #   -> dem is then cropped with intersect so that it lies within intersect
    #       but is not resampled in the process
    #   -> reproject ref to dem's georef-grid,
    #       the intersection grid sampled on dem's grid
    #
    transform_dem = Affine.from_gdal(
        dem["trans"].data[0],
        dem["trans"].data[1],
        dem["trans"].data[2],
        dem["trans"].data[3],
        dem["trans"].data[4],
        dem["trans"].data[5],
    )
    bounds_dem = rasterio.transform.array_bounds(
        dem["im"].data.shape[1], dem["im"].data.shape[0], transform_dem
    )

    transform_ref = Affine.from_gdal(
        ref["trans"].data[0],
        ref["trans"].data[1],
        ref["trans"].data[2],
        ref["trans"].data[3],
        ref["trans"].data[4],
        ref["trans"].data[5],
    )
    bounds_ref = rasterio.transform.array_bounds(
        ref["im"].data.shape[1], ref["im"].data.shape[0], transform_ref
    )

    intersection_roi = (
        max(bounds_dem[0], bounds_ref[0]),
        max(bounds_dem[1], bounds_ref[1]),
        min(bounds_dem[2], bounds_ref[2]),
        min(bounds_dem[3], bounds_ref[3]),
    )

    # get  crop
    polygon_roi = bounding_box_to_polygon(
        intersection_roi[0],
        intersection_roi[1],
        intersection_roi[2],
        intersection_roi[3],
    )
    geom_like_polygon = {"type": "Polygon", "coordinates": [polygon_roi]}

    # crop dem
    srs_dem = rasterio.open(
        " ",
        mode="w+",
        driver="GTiff",
        width=dem["im"].data.shape[1],
        height=dem["im"].data.shape[0],
        count=1,
        dtype=dem["im"].data.dtype,
        crs=dem.attrs["georef"],
        transform=transform_dem,
    )
    srs_dem.write(dem["im"].data, 1)
    new_cropped_dem, new_cropped_dem_transform = rasterio.mask.mask(
        srs_dem, [geom_like_polygon], all_touched=True, crop=True
    )

    # create datasets
    reproj_dem = copy.copy(dem)
    reproj_dem["trans"].data = np.array(new_cropped_dem_transform.to_gdal())
    reproj_dem = read_img_from_array(
        new_cropped_dem[0, :, :],
        from_dataset=reproj_dem,
        no_data=dem.attrs["no_data"],
    )

    # Reference DEM is reprojected into the sec DEM's georef-grid
    # Crop and resample are performed on the reference DEM
    reproj_ref = reproject_dataset(ref, reproj_dem, interp=interpolator)

    return reproj_dem, reproj_ref


def save_tif(
    dataset: xr.Dataset, filename: str, new_array=None, no_data: float = -32768
) -> xr.Dataset:
    """
    Write a Dataset in a tiff file.
    If new_array is set, new_array is used as data.

    :param dataset: dataset
    :param filename:  output filename
    :param new_array:  new array to write
    :param no_data:  value of nodata to use
    :return: dataset
    """

    # update from dataset
    previous_profile = {}
    previous_profile["crs"] = rasterio.crs.CRS.from_dict(
        dataset.attrs["georef"]
    )
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

    new_dataset = copy.deepcopy(dataset)
    # update dataset input_img with new filename
    new_dataset.attrs["input_img"] = filename

    return new_dataset


def get_slope(dataset: xr.Dataset, degree: bool = False) -> np.ndarray:
    """
    Compute slope from dataset
    Slope is presented here :
    http://pro.arcgis.com/ \
            fr/pro-app/tool-reference/spatial-analyst/how-aspect-works.htm

    :param dataset: dataset
    :type dataset: xr.Dataset
    :param degree:  True if is in degree
    :type degree: bool
    :return: slope
    :rtype: np.ndarray
    """

    def get_orthodromic_distance(
        lon1: float, lat1: float, lon2: float, lat2: float
    ):
        """
        Get Orthodromic distance from two (lat,lon) coordinates

        :param lon1:
        :param lat1:
        :param lon2:
        :param lat2:
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

    crs = rasterio.crs.CRS.from_dict(dataset.attrs["georef"])
    if not crs.is_projected:
        # Our dem is not projected, we can't simply use the pixel resolution
        # -> we need to compute resolution between each point
        ny, nx = dataset["im"].data.shape
        xp = np.arange(nx)
        yp = np.arange(ny)
        xp, yp = np.meshgrid(xp, yp)
        lon, lat = pix_to_coord(dataset["trans"].data, yp, xp)
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


def interpolate_geoid(
    geoid_filename: str, coords: np.ndarray, interpol_method: str = "linear"
) -> np.ndarray:
    """
    Bilinear interpolation of geoid

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
    step_x = transform[0]
    # ascending
    step_y = -transform[4]
    # 0 or 0.5

    # coin BG
    [ori_x, ori_y] = transform * (
        0.5,
        dataset.height - 0.5,
    )  # positions au centre pixel

    last_x = ori_x + step_x * dataset.width
    last_y = ori_y + step_y * dataset.height
    # transform dep to positions
    geoid_values = dataset.read(1)[::-1, :].transpose()
    x = np.arange(ori_x, last_x, step_x)
    # lat must be in ascending order,
    y = np.arange(ori_y, last_y, step_y)
    geoid_grid_coordinates = (x, y)
    interp_geoid = interpolate.interpn(
        geoid_grid_coordinates,
        geoid_values,
        coords,
        method=interpol_method,
        bounds_error=False,
        fill_value=None,
    )
    return interp_geoid


def get_geoid_offset(
    dataset: xr.Dataset, geoid_path: Union[str, None]
) -> np.ndarray:
    """
    Get offset from geoid to ellipsoid

    :param dataset: dataset
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

    ny, nx = dataset["im"].data.shape
    xp = np.arange(nx)
    yp = np.arange(ny)

    # xp in [-180, 180], yp in [-90, 90]
    xp[xp > 180] = xp[xp > 180] - 360
    xp[xp < 180] = xp[xp < 180] + 360
    yp[yp > 90] = yp[yp > 90] - 180
    yp[yp < 90] = yp[yp < 90] + 180

    xp, yp = np.meshgrid(xp, yp)

    lon, lat = pix_to_coord(dataset["trans"].data, yp, xp)

    src_crs = rasterio.crs.CRS.from_dict(dataset.attrs["georef"])
    if src_crs.is_projected:
        # convert to global coordinates
        proj = pyproj.Proj(src_crs)
        lon, lat = proj(  # pylint: disable=unpacking-non-sequence
            lon, lat, inverse=True
        )

        # transform to list (2xN)
    lon_1d = np.reshape(
        lon, (dataset["im"].data.shape[0] * dataset["im"].data.shape[1])
    )
    lat_1d = np.reshape(
        lat, (dataset["im"].data.shape[0] * dataset["im"].data.shape[1])
    )
    coords = np.zeros((lon_1d.size, 2))
    coords[:, 0] = lon_1d
    coords[:, 1] = lat_1d

    # Get geoid values
    interp_geoid = interpolate_geoid(
        geoid_path, coords, interpol_method="linear"
    )

    # transform to array of shape dataset['im'].data.shape
    arr_offset = np.reshape(interp_geoid, dataset["im"].data.shape)

    return arr_offset


def compute_offset_bounds(
    y_off: float, x_off: float, cfg: Dict
) -> Tuple[float, float, float, float]:
    """
    Obtain the coordinate bounds to apply the offsets to
    the secondary DEM with gdal

    The offsets can be applied with the command line:
    gdal_translate -a_ullr <ulx> <uly> <lrx> <lry>
    /path_to_original_dem.tif /path_to_coregistered_dem.tif

    :param y_off: y pixel offset
    :type y_off: float
    :param x_off: x pixel offset
    :type x_off: float
    :param cfg: cfg
    :type cfg: Dict
    :return: coordinate bounds to apply the offsets
    :rtype: Tuple[float,float,float,float]
    """
    # Read original secondary dem
    dem = read_img(
        cfg["inputDSM"]["path"],
        load_data=(
            cfg["inputDSM"]["roi"] if "roi" in cfg["inputDSM"] else False
        ),
    )
    ysize, xsize = dem["im"].shape
    # Compute the coordinates of the new bounds
    x_0, y_0 = pix_to_coord(dem["trans"].data, y_off, x_off)
    x_1, y_1 = pix_to_coord(dem["trans"].data, y_off + ysize, x_off + xsize)

    return float(x_0), float(y_0), float(x_1), float(y_1)
