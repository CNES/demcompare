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
Tools to manipulate meshes
"""

import os.path
from typing import Union

# Third party imports
import numpy as np
import open3d as o3d
import pandas as pd
import plyfile

from ..tools import point_cloud_io as pcd_io
from ..tools.handlers import Mesh


def write_triangle_mesh_o3d(
    filepath: str, mesh: Union[dict, Mesh], compressed: bool = True
):
    """Write triangle mesh to disk with open3d"""

    if isinstance(mesh, Mesh):
        mesh.set_o3d_mesh_from_df()
        o3d.io.write_triangle_mesh(
            filepath, mesh.o3d_mesh, compressed=compressed
        )
    else:
        raise NotImplementedError


def serialize_ply_texture(filepath: str, mesh: Mesh) -> None:
    """
    Serialize a textured mesh as a PLY file

    Parameters
    ----------
    filepath: str
        Filepath to the texture image
    mesh: Mesh
        Mesh object
    """
    # Vertices
    vertices = mesh.pcd.get_vertices().to_numpy()
    vertex = np.array(
        list(zip(*vertices.T)), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")]
    )

    # Faces + Texture
    triangles = mesh.get_triangles().to_numpy()
    ply_faces = np.empty(
        len(triangles),
        dtype=[("vertex_indices", "i4", (3,)), ("texcoord", "f8", (6,))],
    )  # 3 pairs of image coordinates
    ply_faces["vertex_indices"] = triangles.astype("i4")

    triangles_uvs = mesh.get_triangle_uvs().to_numpy()
    ply_faces["texcoord"] = triangles_uvs.astype("f8")

    # Define elements
    el_vertex = plyfile.PlyElement.describe(
        vertex, "vertex", val_types={"x": "f8", "y": "f8", "z": "f8"}
    )
    el_vertex_indices = plyfile.PlyElement.describe(
        ply_faces, "face", val_types={"vertex_indices": "i4", "texcoord": "f8"}
    )

    # Mesh and texture image are assumed to be in the same repository
    comments = [f"TextureFile {os.path.basename(mesh.image_texture_path)}"]

    # Write ply file
    plyfile.PlyData(
        [el_vertex, el_vertex_indices], byte_order=">", comments=comments
    ).write(filepath)


# -------------------------------------------------------------------------- #
# Mesh object ===> any mesh format
# -------------------------------------------------------------------------- #


def mesh2ply(filepath: str, mesh: Mesh, compressed: bool = True):
    """Mesh object to PLY mesh"""

    # Check consistency
    if filepath.split(".")[-1] != "ply":
        raise ValueError(
            f"Filepath extension should be '.ply', but found: "
            f"'{filepath.split('.')[-1]}'."
        )

    # # Write point cloud apart in a LAS file
    # filepath_pcd = filepath[:-4] + "_pcd.las"
    # pcd_io.df2las(filepath_pcd, dict_pcd_mesh["pcd"])

    # Write mesh in PLY file
    if mesh.df is not None and mesh.has_texture:
        serialize_ply_texture(filepath, mesh)
    else:
        write_triangle_mesh_o3d(filepath, mesh, compressed=compressed)


# -------------------------------------------------------------------------- #
# any mesh format ===> dict of pandas DataFrame point cloud and numpy array
# mesh (vertex indexes of triangles)
# -------------------------------------------------------------------------- #


def ply2mesh(filepath: str) -> (pd.DataFrame, pd.DataFrame):
    """PLY mesh to Mesh object"""

    # Check consistency
    if filepath.split(".")[-1] != "ply":
        raise ValueError(
            f"Filepath extension should be '.ply', but found: "
            f"'{filepath.split('.')[-1]}'."
        )

    # Read point cloud and faces
    mesh = Mesh(o3d_mesh=o3d.io.read_triangle_mesh(filepath))
    mesh.set_df_from_o3d_mesh()

    return mesh.pcd.df, mesh.df


# -------------------------------------------------------------------------- #
# General functions
# -------------------------------------------------------------------------- #


def deserialize_mesh(filepath: str) -> (pd.DataFrame, pd.DataFrame):
    """Deserialize a mesh"""
    extension = filepath.split(".")[-1]

    if extension == "ply":
        mesh = ply2mesh(filepath)

    else:
        raise NotImplementedError

    return mesh


def serialize_mesh(filepath: str, mesh: Mesh, extension: str = "ply") -> None:
    """Serialize a mesh to disk in the format asked by the user"""

    if filepath.split(".")[-1] != extension:
        raise ValueError(
            f"Filepath extension ('{filepath.split('.')[-1]}') is "
            f"inconsistent with the extension "
            f"asked ('{extension}')."
        )

    if extension == "ply":
        mesh2ply(filepath, mesh)

    elif extension == "csv":
        pcd_io.df2csv(filepath, mesh.df)

    else:
        raise NotImplementedError
