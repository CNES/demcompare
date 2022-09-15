#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Chloe Thenoz (Magellium), Lisa Vo Thanh (Magellium).
#
# This file is part of mesh_3d
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

from typing import Union

import laspy
import numpy as np
import open3d as o3d
import pandas as pd
import plyfile
import pyproj

# LAS tools


def get_offset(arr_max, arr_min):
    return (arr_max + arr_min) / 2


def get_scale(arr_max, arr_min, number_values):
    return (arr_max - arr_min) / number_values


def apply_scale_offset(
    arr, scale, offset, is_inverse=False, clip_min=None, clip_max=None
):
    if not is_inverse:
        # X ==> x
        return np.clip(arr * scale + offset, a_min=clip_min, a_max=clip_max)
    else:
        # x ==> X
        return np.clip((arr - offset) / scale, a_min=clip_min, a_max=clip_max)


# -------------------------------------------------------------------------------------------------------- #
# any point cloud format ===> pandas DataFrame
# -------------------------------------------------------------------------------------------------------- #


def o3d2df(o3d_pcd: o3d.geometry.PointCloud) -> pd.DataFrame:
    """Open3D Point Cloud to pandas DataFrame"""
    from ..tools.handlers import PointCloud

    if not o3d_pcd.is_empty():
        raise ValueError("Open3D Point Cloud does not contain any point.")
    else:
        pcd = PointCloud(o3d_pcd=o3d_pcd)

    # Set df from open3d data
    pcd.set_df_from_o3d_pcd()

    return pcd.df


def las2df(filepath: str) -> pd.DataFrame:
    """LAS or LAZ point cloud to pandas DataFrame"""

    las = laspy.read(filepath)
    dimensions = las.points.array.dtype.names

    df = pd.DataFrame(data=las.xyz, columns=["x", "y", "z"])

    for c in ["red", "green", "blue", "nir", "classification"]:
        if c in dimensions:
            df[c] = las.points.array[c]

    return df


def pkl2df(filepath: str) -> pd.DataFrame:
    """PKL point cloud to pandas DataFrame"""
    return pd.read_pickle(filepath)


def ply2df(filepath: str) -> pd.DataFrame:
    """PLY point cloud to pandas DataFrame"""
    plydata = plyfile.PlyData.read(filepath)

    df = pd.DataFrame()

    for propty in plydata.elements[0].properties:
        name = propty.name
        df[name] = np.array(plydata.elements[0].data[name])

    return df


def csv2df(filepath: str) -> pd.DataFrame:
    """CSV file to pandas DataFrame"""
    return pd.read_csv(filepath)


# -------------------------------------------------------------------------------------------------------- #
# pandas DataFrame ===> any point cloud format
# -------------------------------------------------------------------------------------------------------- #


def df2las(
    filepath: str,
    df: pd.DataFrame,
    metadata: Union[laspy.LasHeader, None] = None,
):
    """
    This method serializes a pandas DataFrame in .las
    """
    if filepath.split(".")[-1] not in ["las", "laz"]:
        raise ValueError(
            "Filepath extension is invalid. It should either be 'las' or 'laz'."
        )

    # Fill header
    header = laspy.LasHeader(point_format=8, version="1.4")

    if metadata is not None:
        header = metadata

    # Compute normalization parameters specific to the LAS format (data is saved as an int32)
    # 32 bits signed ==> 2**31 values (last bit for sign + or -)
    number_values = 2**32

    # x_max = s * X_max + o <==> x_max = 2 ** 31 * s + o
    # x_min = s * X_min + o <==> x_min = - 2 ** 31 * s + o

    header.scales = [
        get_scale(df[coord].max(), df[coord].min(), number_values)
        for coord in ["x", "y", "z"]
    ]
    header.offsets = [
        get_offset(df[coord].max(), df[coord].min())
        for coord in ["x", "y", "z"]
    ]

    # Fill data points
    las = laspy.LasData(header)

    # coords scaled in int32
    [las.X, las.Y, las.Z] = [
        (
            apply_scale_offset(
                df[coord],
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
        if c in df:
            las.points.array[c] = df[c]

    # Write file to disk
    las.write(filepath)


def df2o3d(df_pcd: pd.DataFrame) -> o3d.geometry.PointCloud:
    """pandas.DataFrame to Open3D Point Cloud"""
    from ..tools.handlers import PointCloud

    pcd = PointCloud(df_pcd)
    pcd.set_o3d_pcd_from_df()

    return pcd.o3d_pcd


def df2csv(filepath: str, df: pd.DataFrame, **kwargs):
    """pandas DataFrame to csv file"""

    if filepath.split(".")[-1] != "csv":
        raise ValueError("Filepath extension is invalid. It should be 'csv'.")

    df.to_csv(filepath, index=False, **kwargs)


# -------------------------------------------------------------------------------------------------------- #
# General functions
# -------------------------------------------------------------------------------------------------------- #


def deserialize_point_cloud(filepath: str) -> pd.DataFrame:
    """Convert a point cloud to a pandas dataframe"""
    extension = filepath.split(".")[-1]

    if extension == "las" or extension == "laz":
        df = las2df(filepath)

    elif extension == "pkl":
        df = pkl2df(filepath)

    elif extension == "ply":
        df = ply2df(filepath)

    elif extension == "csv":
        df = csv2df(filepath)

    else:
        raise NotImplementedError

    return df


def serialize_point_cloud(
    filepath: str,
    df: pd.DataFrame,
    metadata: Union[laspy.LasHeader, None] = None,
    extension: str = "las",
):
    """Serialize a point cloud to disk in the format asked by the user"""

    if filepath.split(".")[-1] != extension:
        raise ValueError(
            f"Filepath extension ('{filepath.split('.')[-1]}') is inconsistent with the extension "
            f"asked ('{extension}')."
        )

    if extension == "las" or extension == "laz":
        df2las(filepath, df, metadata=metadata)

    elif extension == "pkl":
        raise NotImplementedError

    elif extension == "ply":
        raise NotImplementedError

    elif extension == "csv":
        df2csv(filepath, df)

    else:
        raise NotImplementedError


def change_frame(df, in_epsg, out_epsg) -> pd.DataFrame:
    """Change frame in which the points are expressed"""
    proj_transformer = pyproj.Transformer.from_crs(
        in_epsg, out_epsg, always_xy=True
    )
    df["x"], df["y"], df["z"] = proj_transformer.transform(
        df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()
    )
    return df


def conversion_utm_to_geo(
    coords: Union[list, tuple, np.ndarray], utm_code: int
):
    """
    Conversion points from epsg 32631 to epsg 4326
    """
    transformer = pyproj.Transformer.from_crs(
        f"epsg:{utm_code}", "epsg:4326", always_xy=True
    )
    out = transformer.transform(coords[:, 0], coords[:, 1], coords[:, 2])

    return np.dstack(out)[0]
