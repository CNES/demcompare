"""
Tools to manipulate point clouds
"""

import laspy
import plyfile
import pandas as pd
import numpy as np


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


def convert_to_pd_dataframe(filepath: str):
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
