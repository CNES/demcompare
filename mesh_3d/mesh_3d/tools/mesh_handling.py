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

import open3d as o3d
import numpy as np

import tools.point_cloud_handling as pcd_handler


# -------------------------------------------------------------------------------------------------------- #
# dict of pandas DataFrame point cloud and numpy array mesh (vertex indexes of triangles) ===> any mesh format
# -------------------------------------------------------------------------------------------------------- #

def dict2o3d(dict_pcd_mesh: dict) -> o3d.geometry.TriangleMesh:
    """dict of pandas DataFrame point cloud and numpy array mesh to open3d Triangle Mesh"""

    # init Triangle Mesh
    mesh = o3d.geometry.TriangleMesh()

    # add vertices
    mesh.vertices = o3d.utility.Vector3dVector(dict_pcd_mesh["pcd"][["x", "y", "z"]].to_numpy())

    # add point indexes forming the triangular faces
    mesh.triangles = o3d.utility.Vector3iVector(dict_pcd_mesh["mesh"].astype(np.int64))

    return mesh


def dict2ply(filepath: str, dict_pcd_mesh: dict):
    """dict of pandas DataFrame point cloud and numpy array mesh to PLY mesh"""

    # Check consistency
    if filepath.split(".")[-1] != "ply":
        raise ValueError(f"Filepath extension should be '.ply', but found: '{filepath.split('.')[-1]}'.")

    # Write point cloud
    if "o3d_pcd" in dict_pcd_mesh:
        o3d.io.write_point_cloud(filepath, dict_pcd_mesh["o3d_pcd"])
    else:
        o3d.io.write_point_cloud(filepath, pcd_handler.df2o3d(dict_pcd_mesh["pcd"]))

    # Write mesh
    if "o3d_mesh" in dict_pcd_mesh:
        o3d.io.write_triangle_mesh(filepath, dict_pcd_mesh["o3d_mesh"])
    else:
        o3d.io.write_triangle_mesh(filepath, dict2o3d(dict_pcd_mesh))


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
    out = {"pcd": pcd_handler.o3d2df(pcd), "mesh": np.asarray(mesh.triangles)}

    return out


# -------------------------------------------------------------------------------------------------------- #
# General functions
# -------------------------------------------------------------------------------------------------------- #

def deserialize_mesh(filepath: str) -> dict:
    """Deserialize a mesh"""
    extension = filepath.split(".")[-1]

    if extension == "ply":
        dict_pcd_mesh = ply2dict(filepath)

    else:
        raise NotImplementedError

    return dict_pcd_mesh


def serialize_mesh(filepath: str,
                   dict_pcd_mesh: dict,
                   extension: str = "ply"):
    """Serialize a mesh to disk in the format asked by the user"""

    if filepath.split(".")[-1] != extension:
        raise ValueError(f"Filepath extension ('{filepath.split('.')[-1]}') is inconsistent with the extension "
                         f"asked ('{extension}').")

    if extension == "ply":
        dict2ply(filepath, dict_pcd_mesh)

    else:
        raise NotImplementedError

