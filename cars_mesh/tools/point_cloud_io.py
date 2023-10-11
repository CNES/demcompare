#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2023 CNES.
#
# This file is part of cars-mesh

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
Tools to manipulate point clouds
"""

# Standard imports
import logging
from typing import Union

# Third party imports
import laspy
import numpy as np
import open3d as o3d
import pandas as pd
import plyfile
import pyproj

# LAS tools


def get_offset(arr_max, arr_min):
    """Compute offset"""
    return (arr_max + arr_min) / 2


def get_scale(arr_max, arr_min, number_values):
    """Compute scale"""
    return (arr_max - arr_min) / number_values


def apply_scale_offset(
    arr: np.ndarray,
    scale: float,
    offset: float,
    is_inverse: bool = False,
    clip_min: Union[float, None] = None,
    clip_max: Union[float, None] = None,
):
    """
    Apply a scale and an offset to the input array

    Parameters
    ----------
    arr: np.ndarray
        Array to normalize
    scale: float
        Scaling factor
    offset: float
        Offset factor
    is_inverse: bool (default=False)
        Whether to denormalize ('inverse') (x = (x' - o) / s) rather than
        normalize (x' = s * x + o)
    clip_min: float or None (default=None)
        Whether to limit the minimum output value
    clip_max: float or None (default=None)
        Whether to limit the maximum output value
    """
    if not is_inverse:
        # x ==> x'
        return np.clip(arr * scale + offset, a_min=clip_min, a_max=clip_max)

    # x' ==> x
    return np.clip((arr - offset) / scale, a_min=clip_min, a_max=clip_max)


# -------------------------------------------------------------------------- #
# any point cloud format ===> pandas DataFrame
# -------------------------------------------------------------------------- #


def o3d2df(o3d_pcd: o3d.geometry.PointCloud) -> pd.DataFrame:
    """Open3D Point Cloud to pandas DataFrame"""
    from ..tools.handlers import PointCloud

    if not o3d_pcd.is_empty():
        raise ValueError("Open3D Point Cloud does not contain any point.")

    pcd = PointCloud(o3d_pcd=o3d_pcd)

    # Set df from open3d data
    pcd.set_df_from_o3d_pcd()

    return pcd.df


def las2df(filepath: str) -> pd.DataFrame:
    """LAS or LAZ point cloud to pandas DataFrame"""

    las = laspy.read(filepath)
    dimensions = las.points.array.dtype.names

    df_pcd = pd.DataFrame(data=las.xyz, columns=["x", "y", "z"])

    for c in ["red", "green", "blue", "nir", "classification"]:
        if c in dimensions:
            df_pcd[c] = las.points.array[c]

    return df_pcd


def pkl2df(filepath: str) -> pd.DataFrame:
    """PKL point cloud to pandas DataFrame"""
    return pd.read_pickle(filepath)


def ply2df(filepath: str) -> pd.DataFrame:
    """PLY point cloud to pandas DataFrame"""
    plydata = plyfile.PlyData.read(filepath)

    df_pcd = pd.DataFrame()

    for propty in plydata.elements[0].properties:
        name = propty.name
        df_pcd[name] = np.array(plydata.elements[0].data[name])

    return df_pcd


def csv2df(filepath: str) -> pd.DataFrame:
    """CSV file to pandas DataFrame"""
    return pd.read_csv(filepath)


# -------------------------------------------------------------------------- #
# pandas DataFrame ===> any point cloud format
# -------------------------------------------------------------------------- #


def df2las(
    filepath: str,
    df_pcd: pd.DataFrame,
    metadata: Union[laspy.LasHeader, None] = None,
    point_format: int = 8,
    version: str = "1.4",
):
    """
    This method serializes a pandas DataFrame in .las
    """
    if filepath.split(".")[-1] not in ["las", "laz"]:
        raise ValueError(
            "Filepath extension is invalid. It should either be 'las' or "
            "'laz'."
        )

    # Fill header
    header = laspy.LasHeader(point_format=point_format, version=version)

    if metadata is not None:
        header = metadata

    # Compute normalization parameters specific to the LAS format (data is
    # saved as an int32)
    # 32 bits signed ==> 2**31 values (last bit for sign + or -)
    number_values = 2**32

    # x_max = s * X_max + o <==> x_max = 2 ** 31 * s + o
    # x_min = s * X_min + o <==> x_min = - 2 ** 31 * s + o

    header.scales = [
        get_scale(df_pcd[coord].max(), df_pcd[coord].min(), number_values)
        for coord in ["x", "y", "z"]
    ]
    header.offsets = [
        get_offset(df_pcd[coord].max(), df_pcd[coord].min())
        for coord in ["x", "y", "z"]
    ]

    # Fill data points
    las = laspy.LasData(header)

    # coords scaled in int32
    [las.X, las.Y, las.Z] = [
        (
            apply_scale_offset(
                df_pcd[coord],
                header.scales[k],
                header.offsets[k],
                is_inverse=True,
                clip_min=-(2**31),
                clip_max=2**31,
            )
        ).astype(np.int32)
        for k, coord in enumerate(["x", "y", "z"])
    ]

    for c in ["red", "green", "blue", "nir", "classification"]:
        if c in df_pcd:
            try:
                las.points.array[c] = df_pcd[c]
            except ValueError:
                logging.warning(
                    f"Field '{c}' is not supported by the point format "
                    f"specified ({point_format}). "
                    f"It will be ignored."
                )

    # Write file to disk
    las.write(filepath)


def df2o3d(df_pcd: pd.DataFrame) -> o3d.geometry.PointCloud:
    """pandas.DataFrame to Open3D Point Cloud"""
    from ..tools.handlers import PointCloud

    pcd = PointCloud(df_pcd)
    pcd.set_o3d_pcd_from_df()

    return pcd.o3d_pcd


def df2csv(filepath: str, df_pcd: pd.DataFrame, **kwargs):
    """pandas DataFrame to csv file"""

    if filepath.split(".")[-1] != "csv":
        raise ValueError("Filepath extension is invalid. It should be 'csv'.")

    df_pcd.to_csv(filepath, index=False, **kwargs)


# -------------------------------------------------------------------------- #
# General functions
# -------------------------------------------------------------------------- #


def deserialize_point_cloud(filepath: str) -> pd.DataFrame:
    """Convert a point cloud to a pandas dataframe"""
    extension = filepath.split(".")[-1]

    if extension in ("las", "laz"):
        df_pcd = las2df(filepath)

    elif extension == "pkl":
        df_pcd = pkl2df(filepath)

    elif extension == "ply":
        df_pcd = ply2df(filepath)

    elif extension == "csv":
        df_pcd = csv2df(filepath)

    else:
        raise NotImplementedError

    return df_pcd


def serialize_point_cloud(
    filepath: str,
    df_pcd: pd.DataFrame,
    metadata: Union[laspy.LasHeader, None] = None,
    extension: str = "las",
    **kwargs,
):
    """Serialize a point cloud to disk in the format asked by the user"""

    if filepath.split(".")[-1] != extension:
        raise ValueError(
            f"Filepath extension ('{filepath.split('.')[-1]}') "
            f"is inconsistent with the extension asked ('{extension}')."
        )

    if extension in ("las", "laz"):
        df2las(filepath, df_pcd, metadata=metadata, **kwargs)

    elif extension == "pkl":
        raise NotImplementedError

    elif extension == "ply":
        raise NotImplementedError

    elif extension == "csv":
        df2csv(filepath, df_pcd)

    else:
        raise NotImplementedError


def change_frame(df_pcd, in_epsg, out_epsg) -> pd.DataFrame:
    """Change frame in which the points are expressed"""
    proj_transformer = pyproj.Transformer.from_crs(
        in_epsg, out_epsg, always_xy=True
    )
    res = proj_transformer.transform(
        df_pcd["x"].to_numpy(), df_pcd["y"].to_numpy(), df_pcd["z"].to_numpy()
    )

    if isinstance(res, tuple) and len(res) == 3:
        df_pcd[["x", "y", "z"]] = np.asarray(res).T
    else:
        raise ValueError(
            f"Something went wrong with the coordinate transform "
            f"process: {res}"
        )

    return df_pcd


def conversion_utm_to_geo(
    coords: Union[list, tuple, np.ndarray], utm_code: int
) -> np.ndarray:
    """
    Conversion points from epsg 32631 to epsg 4326
    """
    transformer = pyproj.Transformer.from_crs(
        f"epsg:{utm_code}", "epsg:4326", always_xy=True
    )
    out = transformer.transform(coords[:, 0], coords[:, 1], coords[:, 2])

    return np.dstack(out)[0]


def convert_color_to_8bits(
    df_pcd: pd.DataFrame, q_percent: Union[tuple, list, np.ndarray] = (0, 100)
) -> pd.DataFrame:
    """
    Convert the colors of the data to 8 bits. It will preserve the relative
    colors between the bands (it is a global normalisation, not a by band
    normalisation).

    Parameters
    ----------
    df_pcd: pd.DataFrame
        Point cloud data
    q_percent: tuple or list or np.ndarray (default=(0, 100))
        Whether to clip the colors to discard outliers. First term is the
        minimum percentage to take into account, the second term is the
        maximum. By default, no value is clipped.

    Returns
    -------
    df_pcd: pd.DataFrame
        Point cloud data with colors converted to 8bits
    """
    from .handlers import COLORS

    # check which color band is in the dataframe
    colors = [c for c in COLORS if c in df_pcd]

    # convert colors to 8bits while preserving the ratio between bands
    # (global normalisation)
    if np.asarray(q_percent).size != 2:
        raise ValueError(
            f"Quantile percentage should be of size 2 (min, max), "
            f"but found here {np.asarray(q_percent).size}."
        )
    q_percent_values = np.percentile(
        df_pcd.loc[:, colors].to_numpy(), q_percent
    )
    arr = np.clip(
        df_pcd.loc[:, colors].to_numpy(),
        a_min=q_percent_values[0],
        a_max=q_percent_values[1],
    )

    # Normalize
    a = 255.0 / (q_percent_values[1] - q_percent_values[0])
    b = -a * q_percent_values[0]

    arr = a * arr + b * np.ones_like(arr)

    # replace in df_pcd
    df_pcd.loc[:, colors] = arr.astype(np.uint8)

    return df_pcd
