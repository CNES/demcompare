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

import numpy as np

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

    def set_df_from_vertices(self, vertices: np.ndarray) -> None:
        """Set point coordinates in the pandas DataFrame"""
        self.df = pd.DataFrame(data=np.asarray(vertices, dtype=np.float64), columns=["x", "y", "z"])

    def set_df_colors(self, colors: np.ndarray, color_names: list) -> None:
        """Set color attributes per point in the pandas DataFrame"""
        colors = np.asarray(colors)

        for c in color_names:
            if c not in COLORS:
                raise ValueError(f"{c} is not a possible color. Should be in {COLORS}.")

        if colors.shape[1] != len(color_names):
            raise ValueError(f"The number of columns ({colors.shape[1]}) is not equal to the number "
                             f"of column names ({len(color_names)}).")

        self.df[color_names] = colors

    def set_df_normals(self, normals: np.ndarray) -> None:
        """Set normal attributes per point in the pandas DataFrame"""
        normals = np.asarray(normals)

        if normals.shape[1] != 3:
            raise ValueError(f"Normals should have three columns (x, y, z). Found ({normals.shape[1]}).")

        self.df[NORMALS] = normals

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
            return "classification" in self.df.head()

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

    def set_df_from_vertices(self, vertices) -> None:
        self.df = pd.DataFrame(data=vertices, columns=["p1", "p2", "p3"])

    def set_o3d_mesh_from_df(self) -> None:
        if self.df is None:
            raise ValueError("Could not set open3d mesh from df mesh because it is empty.")
        if self.pcd.df is None:
            raise ValueError("Could not set open3d mesh from df pcd because it is empty.")

        self.o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.pcd.df[["x", "y", "z"]].to_numpy()),
            triangles=o3d.utility.Vector3iVector(self.df[["p1", "p2", "p3"]].to_numpy())
        )

        # Add attributes if available
        if self.pcd.has_colors:
            self.o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(self.pcd.df[["red", "green", "blue"]].to_numpy())
        if self.pcd.has_normals:
            self.o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(self.pcd.df[NORMALS].to_numpy())
        # TODO: Open3D has no classification attribute: need to do a research in the df pcd to bring them back? the
        #  point order might be different

    def set_df_from_o3d_mesh(self) -> None:
        if self.o3d_mesh is None:
            raise ValueError("Could not set df from open3d mesh because it is empty.")

        # Set face indexes
        self.set_df_from_vertices(np.asarray(self.o3d_mesh.triangles))

        # Set point cloud data
        self.pcd.set_df_from_vertices(np.asarray(self.o3d_mesh.vertices))
        # Add attributes if available
        if self.o3d_mesh.has_vertex_colors():
            self.pcd.set_df_colors(colors=np.asarray(self.o3d_mesh.vertex_colors), color_names=["red", "green", "blue"])
        if self.o3d_mesh.has_vertex_normals():
            self.pcd.set_df_normals(np.asarray(self.o3d_mesh.vertex_normals))
        # TODO: Open3D has no classification attribute: need to do a research in the df pcd to bring them back? the
        #  point order might be different

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
            return "classification" in self.df.head()

    def serialize(self, filepath: str, **kwargs) -> None:
        """Serialize mesh"""
        from .mesh_io import serialize_mesh
        serialize_mesh(filepath, self, **kwargs)

    def deserialize(self, filepath: str) -> None:
        """Deserialize mesh"""
        from .mesh_io import deserialize_mesh
        self.pcd.df, self.df = deserialize_mesh(filepath)
