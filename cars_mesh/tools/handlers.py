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
Define classes for handling common objects
"""

# Standard imports
import logging
import os
from typing import Union

# Third party imports
import numpy as np
import open3d as o3d
import pandas as pd

COLORS = ["red", "green", "blue", "nir"]
NORMALS = ["n_x", "n_y", "n_z"]
UVS = ["uv1_row", "uv1_col", "uv2_row", "uv2_col", "uv3_row", "uv3_col"]


class PointCloud:
    """Point cloud data"""

    def __init__(
        self,
        df: Union[None, pd.DataFrame] = None,
        o3d_pcd: Union[None, o3d.geometry.PointCloud] = None,
    ) -> None:
        if (not isinstance(df, pd.DataFrame)) and (df is not None):
            raise TypeError(
                f"Input point cloud data 'df' should either be None or a "
                f"pd.DataFrame. Here found "
                f"'{type(df)}'."
            )

        if (not isinstance(o3d_pcd, o3d.geometry.PointCloud)) and (
            o3d_pcd is not None
        ):
            raise TypeError(
                f"Input open3d point cloud data 'o3d_pcd' should either be "
                f"None or a o3d.geometry.PointCloud. Here found "
                f"'{type(o3d_pcd)}'."
            )

        self.df = df
        self.o3d_pcd = o3d_pcd

    def set_df_from_o3d_pcd(self):
        """Set pandas.DataFrame from open3D PointCloud"""
        if self.o3d_pcd is None:
            raise ValueError(
                "Could not set df from open3d pcd because it is empty."
            )

        # Set point cloud data
        self.set_df_from_vertices(np.asarray(self.o3d_pcd.vertices))
        # Add attributes if available
        if self.o3d_pcd.has_colors():
            self.set_df_colors(
                colors=np.asarray(self.o3d_pcd.colors),
                color_names=["red", "green", "blue"],
            )
        if self.o3d_pcd.has_normals():
            self.set_df_normals(np.asarray(self.o3d_pcd.normals))
        # TODO: Open3D has no classification attribute: need to do a research
        #  in the df pcd to bring them back? the
        #  point order might be different

    def set_df_from_vertices(self, vertices: np.ndarray) -> None:
        """Set point coordinates in the pandas DataFrame"""
        self.df = pd.DataFrame(
            data=np.asarray(vertices, dtype=np.float64), columns=["x", "y", "z"]
        )

    def set_df_colors(self, colors: np.ndarray, color_names: list) -> None:
        """Set color attributes per point in the pandas DataFrame"""
        colors = np.asarray(colors)

        for c in color_names:
            if c not in COLORS:
                raise ValueError(
                    f"{c} is not a possible color. Should be in {COLORS}."
                )

        if colors.shape[1] != len(color_names):
            raise ValueError(
                f"The number of columns ({colors.shape[1]}) is not "
                f"equal to the number of column names ({len(color_names)})."
            )

        self.df[color_names] = colors

    def set_df_normals(self, normals: np.ndarray) -> None:
        """Set normal attributes per point in the pandas DataFrame"""
        normals = np.asarray(normals)

        if normals.shape[1] != 3:
            raise ValueError(
                f"Normals should have three columns (x, y, z). "
                f"Found ({normals.shape[1]})."
            )

        self.df[NORMALS] = normals

    def set_o3d_pcd_from_df(self):
        """Set open3d PointCloud from pandas.DataFrame"""
        # add np.ascontiguousarray to avoid seg fault in c parts of open3d
        self.o3d_pcd = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(
                np.ascontiguousarray(self.df[["x", "y", "z"]].to_numpy())
            )
        )

        if self.has_colors:
            self.set_o3d_colors()
        if self.has_normals:
            self.set_o3d_normals()

    def set_o3d_colors(self) -> None:
        """Set color attribute of open3D PointCloud"""
        # Check o3d point cloud is initialized
        if self.o3d_pcd is None:
            raise ValueError("Open3D Point Cloud is empty.")

        # add colors if applicable (only RGB)
        # init to zero
        colors_arr = np.zeros_like(
            self.df[["x", "y", "z"]].to_numpy(), dtype=np.float64
        )
        # retrieve information from the dataframe
        for k, c in enumerate(["red", "green", "blue"]):
            if c in self.df:
                colors_arr[:, k] = self.df[c].to_numpy()
            else:
                raise ValueError(
                    f"Open3D only deals with RGB colors. Here '{c}' is "
                    f"missing."
                )
        # normalize colours in [0, 1]
        colors_arr = np.divide(
            colors_arr - colors_arr.min(),
            colors_arr.max() - colors_arr.min(),
            out=np.zeros_like(colors_arr),
            where=(colors_arr.max() - colors_arr.min()) != 0.0,
        )
        # add to opend3d point cloud
        self.o3d_pcd.colors = o3d.utility.Vector3dVector(
            np.ascontiguousarray(colors_arr)
        )

    def set_o3d_normals(self) -> None:
        """Set normal attribute of open3D PointCloud"""

        # Check o3d point cloud is initialized
        if self.o3d_pcd is None:
            raise ValueError("Open3D Point Cloud is empty.")

        self.o3d_pcd.normals = o3d.utility.Vector3dVector(
            np.ascontiguousarray(self.df[NORMALS].to_numpy())
        )

    def get_vertices(self) -> pd.DataFrame:
        """Get vertex data"""
        return self.df[["x", "y", "z"]]

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

    def set_unitary_normals(self):
        """Make normals unitary (i.e. with a norm equal to 1.)"""
        if not self.has_normals:
            raise ValueError("Point cloud has no normals.")

        self.df[NORMALS] /= np.linalg.norm(
            self.df[NORMALS].to_numpy(), axis=1, keepdims=True
        )

    @property
    def has_colors(self) -> bool:
        """Whether colors are specified"""
        if self.df is None:
            raise ValueError("Point cloud (pandas DataFrame) is not assigned.")

        return any(c in self.df.head() for c in COLORS)

    @property
    def has_normals(self) -> bool:
        """Whether point normals are specified"""
        if self.df is None:
            raise ValueError("Point cloud (pandas DataFrame) is not assigned.")

        return all(n in self.df.head() for n in NORMALS)

    @property
    def are_normals_unitary(self) -> bool:
        """Whether normals are unitary (i.e. of norm equal to 1)"""
        if self.df is None:
            raise ValueError("Point cloud (pandas DataFrame) is not assigned.")

        return self.has_normals and np.all(
            np.equal(
                # Results are rounded in order to avoid taking into
                # account numerical errors
                np.around(
                    np.linalg.norm(self.df[NORMALS].to_numpy(), axis=1),
                    decimals=9,
                ),
                np.ones(self.df.shape[0], dtype=np.float64),
            )
        )

    @property
    def has_classes(self) -> bool:
        """Whether point cloud has classes specified by point"""
        if self.df is None:
            raise ValueError("Point cloud (pandas DataFrame) is not assigned.")

        return "classification" in self.df.head()

    def serialize(self, filepath: str, **kwargs) -> None:
        """Serialize point cloud"""
        from .point_cloud_io import serialize_point_cloud

        serialize_point_cloud(filepath, self.df, **kwargs)

    def deserialize(self, filepath: str) -> None:
        """Deserialize point cloud"""
        from .point_cloud_io import deserialize_point_cloud

        self.df = deserialize_point_cloud(filepath)


class Mesh:
    """Mesh data"""

    def __init__(
        self,
        pcd: Union[None, pd.DataFrame] = None,
        mesh: Union[None, pd.DataFrame] = None,
        o3d_pcd: Union[None, o3d.geometry.PointCloud] = None,
        o3d_mesh: Union[None, o3d.geometry.TriangleMesh] = None,
    ):
        self.pcd = PointCloud(df=pcd, o3d_pcd=o3d_pcd)
        self.df = mesh

        self.o3d_mesh = o3d_mesh

        # image texture
        self.image_texture_path = None

    def set_df_from_o3d_mesh(self) -> None:
        """Set pd.DataFrame from open3d TriangleMesh data"""
        if self.o3d_mesh is None:
            raise ValueError(
                "Could not set df from open3d mesh because it is empty."
            )

        # Set face indexes
        self.set_df_from_vertices(np.asarray(self.o3d_mesh.triangles))

        # Set point cloud data
        self.pcd.set_df_from_vertices(np.asarray(self.o3d_mesh.vertices))

        # Add attributes if available
        # Mesh
        if self.o3d_mesh.has_textures():
            if self.image_texture_path is None:
                logging.warning(
                    "No image texture path is given to the Mesh object. "
                    "Texture will remain incomplete."
                )

        if self.o3d_mesh.has_triangle_uvs():
            self.set_df_uvs(
                uvs=np.asarray(self.o3d_mesh.triangle_uvs).reshape((-1, 6))
            )

        # Point Cloud
        if self.o3d_mesh.has_vertex_colors():
            self.pcd.set_df_colors(
                colors=np.asarray(self.o3d_mesh.vertex_colors),
                color_names=["red", "green", "blue"],
            )
        if self.o3d_mesh.has_vertex_normals():
            self.pcd.set_df_normals(np.asarray(self.o3d_mesh.vertex_normals))
        # TODO: Open3D has no classification attribute: need to do a research
        #  in the df pcd to bring them back? the
        #  point order might be different

    def set_df_from_vertices(self, vertices) -> None:
        """Set pd.DataFrame from an array of vertices"""
        self.df = pd.DataFrame(data=vertices, columns=["p1", "p2", "p3"])

    def set_image_texture_path(self, image_texture_path) -> None:
        """Set image texture path"""
        self.image_texture_path = image_texture_path

    def set_df_uvs(self, uvs) -> None:
        """
        UVs

        Parameters
        ----------
        uvs: (N, 6) np.ndarray or list
            Image texture (row, col) normalized coordinates per triangle vertex
        """
        uvs = np.asarray(uvs)

        # Check characteristics
        if uvs.shape[0] != self.df.shape[0]:
            raise ValueError(
                f"Inconsistent number of triangles between triangle vertex "
                f"indexes ({self.df.shape[0]} "
                f"triangles) and uvs ({uvs.shape[0]} data)."
            )
        if uvs.shape[1] != 6:
            raise ValueError(
                f"UVs should be a numpy ndarray or a list of list with "
                f"exactly 6 columns (3 vertices associated to a pair of "
                f"image texture coordinates (row, col). "
                f"Here found {uvs.shape[1]}."
            )

        self.df[UVS] = uvs

    def set_o3d_mesh_from_df(self) -> None:
        """Set open3d TriangleMesh from a pd.DataFrame"""
        if self.df is None:
            raise ValueError(
                "Could not set open3d mesh from df mesh because it is empty."
            )
        if self.pcd.df is None:
            raise ValueError(
                "Could not set open3d mesh from df pcd because it is empty."
            )

        self.o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(
                np.ascontiguousarray(self.pcd.df[["x", "y", "z"]].to_numpy())
            ),
            triangles=o3d.utility.Vector3iVector(
                np.ascontiguousarray(self.df[["p1", "p2", "p3"]].to_numpy())
            ),
        )
        # Add attributes if available
        # Mesh
        if self.has_texture:
            self.set_o3d_image_texture_and_uvs()
        # Point cloud
        if self.pcd.has_colors:
            self.set_o3d_vertex_colors()
        if self.pcd.has_normals:
            self.set_o3d_vertex_normals()

        # TODO: Open3D has no classification attribute: need to do a research
        #  in the df pcd to bring them back? the point order might be different

    def set_o3d_vertex_colors(self) -> None:
        """Set color attribute of open3D TriangleMesh"""

        # Check o3d mesh is initialized
        if self.o3d_mesh is None:
            raise ValueError(
                "Could not set df from open3d mesh because it is empty."
            )

        # add colors if applicable (only RGB)
        # init to zero
        colors_arr = np.zeros_like(
            self.pcd.df[["x", "y", "z"]].to_numpy(), dtype=np.float64
        )
        # retrieve information from the dataframe
        for k, c in enumerate(["red", "green", "blue"]):
            if c in self.pcd.df:
                colors_arr[:, k] = self.pcd.df[c].to_numpy()
            else:
                raise ValueError(
                    f"Open3D only deals with RGB colors. Here '{c}' is "
                    f"missing."
                )
        # normalize colours in [0, 1]
        colors_arr = np.divide(
            colors_arr - colors_arr.min(),
            colors_arr.max() - colors_arr.min(),
            out=np.zeros_like(colors_arr),
            where=(colors_arr.max() - colors_arr.min()) != 0.0,
        )
        # add to opend3d mesh
        self.o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.ascontiguousarray(colors_arr)
        )

    def set_o3d_vertex_normals(self) -> None:
        """Set normal attribute of open3D TriangleMesh"""

        # Check o3d mesh is initialized
        if self.o3d_mesh is None:
            raise ValueError(
                "Could not set df from open3d mesh because it is empty."
            )

        self.o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(
            np.ascontiguousarray(self.pcd.df[NORMALS].to_numpy())
        )

    def set_o3d_image_texture_and_uvs(self) -> None:
        """Set image texture path and uvs of open3D TriangleMesh"""

        # Check o3d mesh is initialized
        if self.o3d_mesh is None:
            raise ValueError(
                "Could not set df from open3d mesh because it is empty."
            )

        if not self.has_texture:
            raise ValueError(
                "Mesh object has no texture (either the image texture path "
                "or the uvs are missing."
            )

        # UVs in open3d are expressed as a (3 * num_triangles, 2)
        # Reshape data before feeding open3d TriangleMesh
        uvs = self.df[UVS].to_numpy()
        uvs = uvs.reshape((-1, 2))
        self.o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)

        # Add image texture path
        self.o3d_mesh.textures = [o3d.io.read_image(self.image_texture_path)]

    def get_triangles(self) -> pd.DataFrame:
        """Get point triangle indexes"""
        return self.df[["p1", "p2", "p3"]]

    def get_triangle_uvs(self) -> pd.DataFrame:
        """Get triangle uvs"""
        if not self.has_triangle_uvs:
            raise ValueError("Mesh has no triangle uvs.")

        return self.df[UVS]

    def get_image_texture_path(self) -> str:
        """Get image texture path"""
        if self.image_texture_path is None:
            raise ValueError("Mesh has no image texture path defined.")

        return self.image_texture_path

    @property
    def has_triangles(self) -> bool:
        """Whether mesh has triangles specified"""
        if self.df is None:
            raise ValueError("Mesh (pandas DataFrame) is not assigned.")

        return (
            all(n in self.df.head() for n in ["p1", "p2", "p3"])
            and not self.df.empty
        )

    @property
    def has_texture(self) -> bool:
        """Whether mesh has a texture specified"""
        if self.df is None:
            raise ValueError("Mesh (pandas DataFrame) is not assigned.")

        return (
            (self.image_texture_path is not None)
            and all(
                el in self.df.head()
                for el in [
                    "uv1_row",
                    "uv1_col",
                    "uv2_row",
                    "uv2_col",
                    "uv3_row",
                    "uv3_col",
                ]
            )
            and not self.df.empty
        )

    @property
    def has_triangle_uvs(self) -> bool:
        """Whether mesh has triangle uvs"""
        if self.df is None:
            raise ValueError("Mesh (pandas DataFrame) is not assigned.")

        return (
            all(
                el in self.df.head()
                for el in [
                    "uv1_row",
                    "uv1_col",
                    "uv2_row",
                    "uv2_col",
                    "uv3_row",
                    "uv3_col",
                ]
            )
            and not self.df.empty
        )

    @property
    def has_normals(self) -> bool:
        """Whether mesh has face normals"""
        if self.df is None:
            raise ValueError("Mesh (pandas DataFrame) is not assigned.")

        return all(n in self.df.head() for n in NORMALS) and not self.df.empty

    @property
    def has_classes(self) -> bool:
        """Whether mesh has a classification by face"""
        if self.df is None:
            raise ValueError("Mesh (pandas DataFrame) is not assigned.")

        return "classification" in self.df.head() and not self.df.empty

    def serialize(self, filepath: str, **kwargs) -> None:
        """Serialize mesh"""
        from .mesh_io import serialize_mesh

        serialize_mesh(filepath, self, **kwargs)

    def deserialize(self, filepath: str) -> None:
        """Deserialize mesh"""
        from .mesh_io import deserialize_mesh

        self.pcd.df, self.df = deserialize_mesh(filepath)


def read_input_path(input_path: str) -> Mesh:
    """
    Read input path as either a PointCloud or a Mesh object
    and returns in generic Mesh() object (containing pcd dict if point cloud)

    Parameters
    ----------
    input_path: str
        Input path to read (best with absolute path)

    Returns
    -------
    mesh: Mesh
        Mesh object
    """
    from ..param import MESH_FILE_EXTENSIONS

    if os.path.basename(input_path).split(".")[-1] in MESH_FILE_EXTENSIONS:
        # Try reading input data as a mesh if the extension is valid
        mesh = Mesh()
        mesh.deserialize(input_path)
        logging.debug("Input data read as a mesh format.")

    else:
        # If the extension is not a mesh extension, read the data as a
        # point cloud and put it in a dict in Mesh structure
        mesh = Mesh()
        mesh.pcd.deserialize(input_path)
        logging.debug("Input data read as a point cloud format.")

    return mesh
