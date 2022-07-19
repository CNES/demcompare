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
Define classes for handling common objects
"""

from typing import Union

import pandas as pd
import open3d as o3d


COLORS = ["red", "green", "blue", "nir"]
NORMALS = ["n_x", "n_y", "n_z"]


class PointCloud(object):
    """Point cloud data"""
    def __init__(self,
                 df: Union[None, pd.DataFrame] = None,
                 o3d_pcd: Union[None, o3d.geometry.PointCloud] = None) -> None:

        if (not isinstance(df, pd.DataFrame)) and (df is not None):
            raise TypeError(f"Input point cloud data 'df' should either be None or a pd.DataFrame. Here found "
                            f"'{type(df)}'.")

        if (not isinstance(o3d_pcd, o3d.geometry.PointCloud)) and (o3d_pcd is not None):
            raise TypeError(f"Input open3d point cloud data 'o3d_pcd' should either be None or a "
                            f"o3d.geometry.PointCloud. Here found '{type(o3d_pcd)}'.")

        self.df = df
        self.o3d_pcd = o3d_pcd

    def get_colors(self) -> pd.DataFrame:
        """Get color data"""
        if not self.has_colors:
            raise ValueError("Point cloud has no color.")
        return self.df[[c for c in COLORS if c in self.df.head()]]

    def get_normals(self) -> pd.DataFrame:
        """Get normals"""
        if not self.has_normals:
            raise ValueError("Point cloud has no normals.")
        return self.df[NORMALS]

    @property
    def has_colors(self) -> bool:
        if self.df is None:
            raise ValueError("Point cloud (pandas DataFrame) is not assigned.")
        else:
            return any([c in self.df.head() for c in COLORS])

    @property
    def has_normals(self) -> bool:
        if self.df is None:
            raise ValueError("Point cloud (pandas DataFrame) is not assigned.")
        else:
            return all([n in self.df.head() for n in NORMALS])

    @property
    def has_classes(self) -> bool:
        if self.df is None:
            raise ValueError("Point cloud (pandas DataFrame) is not assigned.")
        else:
            return "class" in self.df.head()

    def serialize(self, filepath: str, **kwargs) -> None:
        """Serialize point cloud"""
        from .point_cloud_io import serialize_point_cloud
        serialize_point_cloud(filepath, self.df, **kwargs)

    def deserialize(self, filepath: str) -> None:
        """Deserialize point cloud"""
        from .point_cloud_io import deserialize_point_cloud
        self.df = deserialize_point_cloud(filepath)


class Mesh(object):
    """Mesh data"""
    def __init__(self,
                 pcd: Union[None, pd.DataFrame] = None,
                 mesh: Union[None, pd.DataFrame] = None,
                 o3d_pcd: Union[None, o3d.geometry.PointCloud] = None,
                 o3d_mesh: Union[None, o3d.geometry.TriangleMesh] = None):

        self.pcd = PointCloud(df=pcd, o3d_pcd=o3d_pcd)
        self.df = mesh

        self.o3d_mesh = o3d_mesh

    def set_df_from_vertices(self, vertices):
        self.df = pd.DataFrame(data=vertices, columns=["p1", "p2", "p3"])

    @property
    def has_texture(self) -> bool:
        raise NotImplementedError

    @property
    def has_normals(self) -> bool:
        if self.df is None:
            raise ValueError("Mesh (pandas DataFrame) is not assigned.")
        else:
            return all([n in self.df.head() for n in NORMALS])

    @property
    def has_classes(self) -> bool:
        if self.df is None:
            raise ValueError("Mesh (pandas DataFrame) is not assigned.")
        else:
            return "class" in self.df.head()

    def serialize(self, filepath: str, **kwargs) -> None:
        """Serialize mesh"""
        from .mesh_io import serialize_mesh
        serialize_mesh(filepath, self, **kwargs)

    def deserialize(self, filepath: str) -> None:
        """Deserialize mesh"""
        from .mesh_io import deserialize_mesh
        deserialize_mesh(filepath)
