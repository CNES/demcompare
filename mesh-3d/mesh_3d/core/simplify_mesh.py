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
Simplifying methods for the mesh to decrease the number of faces
"""

import open3d as o3d
import pandas as pd

from ..tools.point_cloud_io import df2o3d
from ..tools.handlers import Mesh, PointCloud


def simplify_quadric_decimation(mesh: Mesh,
                                target_number_of_triangles: int,
                                maximum_error: float = float("inf"),
                                boundary_weight: float = 1.) -> Mesh:
    """
    Function to simplify mesh using Quadric Error Metric Decimation by Garland and Heckbert

    Parameters
    ----------
    mesh: Mesh
        Mesh object
    target_number_of_triangles: int
        The number of triangles that the simplified mesh should have. It is not guaranteed that this number will be reached.
    maximum_error: float (default=inf)
        The maximum error where a vertex is allowed to be merged
    boundary_weight: float (default=1.0)
        A weight applied to edge vertices used to preserve boundaries

    Returns
    -------
    out_mesh: Mesh
        Simplified mesh object
    """

    # Create an open3d mesh if not present
    if mesh.o3d_mesh is None:
        mesh.set_o3d_mesh_from_df()

    # Simplify
    out_mesh_o3d = mesh.o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(target_number_of_triangles),
        maximum_error=float(maximum_error),
        boundary_weight=float(boundary_weight))

    # Create a new mesh object
    out_mesh = Mesh(o3d_mesh=out_mesh_o3d)
    out_mesh.set_df_from_o3d_mesh()

    return out_mesh


def simplify_vertex_clustering(mesh: Mesh, voxel_size: float,
                               contraction: o3d.geometry.SimplificationContraction = o3d.geometry.SimplificationContraction.Average) -> Mesh:
    """
    Function to simplify mesh using vertex clustering.

    Parameters
    ----------
    mesh: Mesh
        Mesh object
    voxel_size: float
        The size of the voxel within vertices are pooled.
    contraction: open3d.geometry.SimplificationContraction (default=<SimplificationContraction.Average – 0>)
        Method to aggregate vertex information. Average computes a simple average, Quadric minimizes the distance
        to the adjacent planes.

    Returns
    -------
    out_mesh: Mesh
        Simplified mesh object
    """

    # Create an open3d mesh if not present
    if mesh.o3d_mesh is None:
        mesh.set_o3d_mesh_from_df()

    # Simplify
    out_mesh_o3d = mesh.o3d_mesh.simplify_vertex_clustering(voxel_size=float(voxel_size),
                                                            contraction=contraction)

    # Create a new mesh object
    out_mesh = Mesh(o3d_mesh=out_mesh_o3d)
    out_mesh.set_df_from_o3d_mesh()

    return mesh
