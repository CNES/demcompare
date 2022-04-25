"""
Tools to manipulate point clouds
"""
from typing import Union
import laspy
import plyfile
import pandas as pd
import numpy as np
import pyproj


def las2df(filepath: str):
    """LAS or LAZ point cloud to pandas DataFrame"""
    las = laspy.read(filepath)
    metadata = las.header

    df = pd.DataFrame.from_records(las.points.array)
    df["X"] = las.xyz[:, 0]
    df["Y"] = las.xyz[:, 1]
    df["Z"] = las.xyz[:, 2]

    df.rename(columns={"X": "x", "Y": "y", "Z": "z"}, inplace=True)

    return df, metadata


def pkl2df(filepath: str):
    """PKL point cloud to pandas DataFrame"""
    return pd.read_pickle(filepath)


def ply2df(filepath: str):
    """PLY point cloud to pandas DataFrame"""
    plydata = plyfile.PlyData.read(filepath)

    df = pd.DataFrame()

    for propty in plydata.elements[0].properties:
        name = propty.name
        df[name] = np.array(plydata.elements[0].data[name])

    return df


def df2las(filepath: str, df: pd.DataFrame, metadata: Union[laspy.LasHeader, None] = None):
    """
    This method serializes a pandas DataFrame in .las
    """
    if filepath.split(".")[-1] not in ["las", "laz"]:
        raise ValueError("Filepath extension is invalid. It should either be 'las' or 'laz'.")

    # Fill header
    header = laspy.LasHeader(point_format=8, version="1.4")

    if metadata is not None:
        header = metadata

    # Compute normalization parameters specific to the LAS format (data is saved as an int32)
    # 32 bits signed ==> 2**31 values (last bit for sign + or -)
    number_values = 2 ** 32

    # x_max = s * X_max + o <==> x_max = 2 ** 31 * s + o
    # x_min = s * X_min + o <==> x_min = - 2 ** 31 * s + o

    def get_offset(arr_max, arr_min):
        return (arr_max + arr_min) / 2

    def get_scale(arr_max, arr_min):
        return (arr_max - arr_min) / number_values

    def apply_scale_offset(arr, scale, offset, is_inverse=False, clip_min=None, clip_max=None):
        if not is_inverse:
            # X ==> x
            return np.clip(arr * scale + offset, a_min=clip_min, a_max=clip_max)
        else:
            # x ==> X
            return np.clip((arr - offset) / scale, a_min=clip_min, a_max=clip_max)

    header.scales = [get_scale(df[coord].max(), df[coord].min()) for coord in ["x", "y", "z"]]
    header.offsets = [get_offset(df[coord].max(), df[coord].min()) for coord in ["x", "y", "z"]]

    # Fill data points
    las = laspy.LasData(header)

    # coords scaled in int32
    [las.X, las.Y, las.Z] = [(apply_scale_offset(df[coord],
                                                 header.scales[k],
                                                 header.offsets[k],
                                                 is_inverse=True,
                                                 clip_min=- 2 ** 31,
                                                 clip_max=2 ** 31)).astype(np.int32)
                             for k, coord in enumerate(["x", "y", "z"])]

    # if there are colors assigned
    if "red" in df:
        las.red = df["red"]
    if "green" in df:
        las.green = df["green"]
    if "blue" in df:
        las.blue = df["blue"]
    if "nir" in df:
        las.nir = df["nir"]

    # Write file to disk
    las.write(filepath)


def deserialize_point_cloud(filepath: str):
    """Convert a point cloud to a pandas dataframe"""
    extension = filepath.split(".")[-1]

    if extension == "las" or extension == "laz":
        df, metadata = las2df(filepath)

    elif extension == "pkl":
        metadata = None
        df = pkl2df(filepath)

    elif extension == "ply":
        metadata = None
        df = ply2df(filepath)

    else:
        raise NotImplementedError

    return df, metadata


def serialize_point_cloud(filepath: str,
                          df: pd.DataFrame,
                          metadata: Union[laspy.LasHeader, None] = None,
                          extension: str = "las"):
    """Serialize a point cloud to disk in the format asked by the user"""

    if filepath.split(".")[-1] != extension:
        raise ValueError(f"Filepath extension ('{filepath.split('.')[-1]}') is inconsistent with the extension "
                         f"asked ('{extension}').")

    if extension == "las" or extension == "laz":
        df2las(filepath, df, metadata=metadata)

    elif extension == "pkl":
        raise NotImplementedError

    elif extension == "ply":
        raise NotImplementedError

    else:
        raise NotImplementedError


def change_frame(df, in_epsg, out_epsg):
    """Change frame in which the points are expressed"""
    proj_transformer = pyproj.Transformer.from_crs(in_epsg, out_epsg, always_xy=True)
    df["x"], df["y"], df["z"] = proj_transformer.transform(df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy())
    return df
