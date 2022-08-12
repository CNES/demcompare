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
Tools to manipulate meshes
"""
import os.path
from typing import Union

import open3d as o3d
import numpy as np
import pandas as pd

from ..tools import point_cloud_io as pcd_io
from ..tools.handlers import Mesh
import plyfile


def write_triangle_mesh_o3d(filepath: str, mesh: Union[dict, Mesh], compressed: bool = True):
    """Write triangle mesh to disk with open3d"""

    # # Write mesh
    # if "o3d_mesh" in dict_pcd_mesh:
    #     o3d.io.write_triangle_mesh(filepath, dict_pcd_mesh["o3d_mesh"], compressed=compressed)
    # else:
    if isinstance(mesh, dict):
        o3d.io.write_triangle_mesh(filepath, dict2o3d(mesh), compressed=compressed)
    elif isinstance(mesh, Mesh):
        mesh.set_o3d_mesh_from_df()
        o3d.io.write_triangle_mesh(filepath, mesh.o3d_mesh, compressed=compressed)
    else:
        raise NotImplementedError

def serialize_ply_texture(filepath: str, mesh: Mesh):

    # Vertices
    vertices = mesh.pcd.df[["x", "y", "z"]].to_numpy()
    vertex = np.array(list(zip(*vertices.T)),
                      dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])

    # Faces + Texture
    triangles = mesh.df[["p1", "p2", "p3"]].to_numpy()
    ply_faces = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,)),
                                                ('texcoord', 'f8', (6,))])  # 3 pairs of image coordinates
    ply_faces['vertex_indices'] = triangles.astype('i4')

    triangles_uvs = mesh.df[["uv1_row", "uv1_col", "uv2_row", "uv2_col", "uv3_row", "uv3_col"]].to_numpy()
    ply_faces['texcoord'] = triangles_uvs.astype('f8')

    # Define elements
    el_vertex = plyfile.PlyElement.describe(vertex, 'vertex', val_types={'x': 'f8', 'y': 'f8', 'z': 'f8'})
    el_vertex_indices = plyfile.PlyElement.describe(ply_faces, 'face', val_types={'vertex_indices': 'i4',
                                                                                  'texcoord': 'f8'})

    comments = [f'TextureFile {os.path.basename(mesh.path_img)}']

    # Write ply file
    plyfile.PlyData([el_vertex, el_vertex_indices], byte_order='>', comments=comments).write(filepath)

# -------------------------------------------------------------------------------------------------------- #
# Mesh object ===> any mesh format
# -------------------------------------------------------------------------------------------------------- #

def mesh2ply(filepath: str, mesh: Mesh, compressed: bool = True):
    """Mesh object to PLY mesh"""

    # Check consistency
    if filepath.split(".")[-1] != "ply":
        raise ValueError(f"Filepath extension should be '.ply', but found: '{filepath.split('.')[-1]}'.")

    # # Write point cloud apart in a LAS file
    # filepath_pcd = filepath[:-4] + "_pcd.las"
    # pcd_io.df2las(filepath_pcd, dict_pcd_mesh["pcd"])

    # Write mesh in PLY file
    filepath_mesh = filepath[:-4] + "_mesh.ply"
    if mesh.has_texture:
        serialize_ply_texture(filepath_mesh, mesh)
    else:
        write_triangle_mesh_o3d(filepath_mesh, mesh, compressed=compressed)

# -------------------------------------------------------------------------------------------------------- #
# dict of pandas DataFrame point cloud and numpy array mesh (vertex indexes of triangles) ===> any mesh format
# -------------------------------------------------------------------------------------------------------- #


def dict2o3d(dict_pcd_mesh: dict) -> o3d.geometry.TriangleMesh:
    """dict of pandas DataFrame point cloud and numpy array mesh to open3d Triangle Mesh"""

    # init Triangle Mesh
    mesh = o3d.geometry.TriangleMesh()

    # add vertices
    mesh.vertices = o3d.utility.Vector3dVector(dict_pcd_mesh["pcd"][["x", "y", "z"]].to_numpy())

    # add colors (if applicable)
    # TODO: implement
    # is_color = [False] * 4
    # colors = ["red", "green", "blue", "nir"]
    # for k, c in enumerate(colors):
    #     if c in dict_pcd_mesh["pcd"]:
    #         is_color[k] = True

    # colors need to be in [0, 1] and is 3-channel

    # add normals
    # TODO: implement

    # add point indexes forming the triangular faces
    mesh.triangles = o3d.utility.Vector3iVector(dict_pcd_mesh["mesh"].astype(np.int64))

    return mesh


def dict2ply(filepath: str, dict_pcd_mesh: dict, compressed: bool = True):
    """dict of pandas DataFrame point cloud and numpy array mesh to PLY mesh"""

    # Check consistency
    if filepath.split(".")[-1] != "ply":
        raise ValueError(f"Filepath extension should be '.ply', but found: '{filepath.split('.')[-1]}'.")

    # # Write point cloud apart in a LAS file
    # filepath_pcd = filepath[:-4] + "_pcd.las"
    # pcd_io.df2las(filepath_pcd, dict_pcd_mesh["pcd"])

    # Write mesh in PLY file
    filepath_mesh = filepath[:-4] + "_mesh.ply"
    write_triangle_mesh_o3d(filepath_mesh, dict_pcd_mesh, compressed=compressed)


def dict2obj(filepath: str, dict_pcd_mesh: dict, compressed: bool = True):
    """dict of pandas DataFrame point cloud and numpy array mesh to OBJ mesh"""

    # TODO: Regularize the point cloud which is not desirable

    # Check consistency
    if filepath.split(".")[-1] != "obj":
        raise ValueError(f"Filepath extension should be '.obj', but found: '{filepath.split('.')[-1]}'.")

    # Write mesh
    write_triangle_mesh_o3d(filepath, dict_pcd_mesh, compressed=compressed)


# -------------------------------------------------------------------------------------------------------- #
# any mesh format ===> dict of pandas DataFrame point cloud and numpy array mesh (vertex indexes of triangles)
# -------------------------------------------------------------------------------------------------------- #

def ply2dict(filepath: str) -> dict:
    """PLY mesh to dict"""

    # Check consistency
    if filepath.split(".")[-1] != "ply":
        raise ValueError(f"Filepath extension should be '.ply', but found: '{filepath.split('.')[-1]}'.")

    # Read point cloud
    pcd = o3d.io.read_point_cloud(filepath)

    # Read mesh
    mesh = o3d.io.read_triangle_mesh(filepath)

    # Convert to df for pcd and numpy array for mesh
    out = {"pcd": pcd_io.o3d2df(pcd), "mesh": np.asarray(mesh.triangles)}

    return out


def ply2mesh(filepath: str) -> (pd.DataFrame, pd.DataFrame):
    """PLY mesh to Mesh object"""

    # Check consistency
    if filepath.split(".")[-1] != "ply":
        raise ValueError(f"Filepath extension should be '.ply', but found: '{filepath.split('.')[-1]}'.")

    # Read point cloud
    df_pcd = pcd_io.deserialize_point_cloud(filepath)

    # Read faces
    df_mesh = pd.DataFrame(data=np.asarray(o3d.io.read_triangle_mesh(filepath).triangles), columns=["p1", "p2", "p3"])

    return df_pcd, df_mesh


# -------------------------------------------------------------------------------------------------------- #
# General functions
# -------------------------------------------------------------------------------------------------------- #

def deserialize_mesh(filepath: str) -> Mesh:
    """Deserialize a mesh"""
    extension = filepath.split(".")[-1]

    if extension == "ply":
        mesh = ply2mesh(filepath)

    else:
        raise NotImplementedError

    return mesh


def serialize_mesh(filepath: str,
                   mesh: Mesh,
                   extension: str = "ply") -> None:
    """Serialize a mesh to disk in the format asked by the user"""

    if filepath.split(".")[-1] != extension:
        raise ValueError(f"Filepath extension ('{filepath.split('.')[-1]}') is inconsistent with the extension "
                         f"asked ('{extension}').")

    if extension == "ply":
        mesh2ply(filepath, mesh)

    elif extension == "csv":
        pcd_io.df2csv(filepath, mesh.df)

    else:
        raise NotImplementedError

