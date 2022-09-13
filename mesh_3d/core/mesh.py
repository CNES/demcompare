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
Meshing methods to create a surface from the point cloud.
"""

from typing import Union

import matplotlib.tri as mtri
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import Delaunay

from ..tools.handlers import Mesh, PointCloud
from ..tools.point_cloud_io import df2o3d


def ball_pivoting_reconstruction(
    pcd: PointCloud, radii: Union[list, float, None] = 0.6
) -> Mesh:
    """
    Bernardini, Fausto et al. “The ball-pivoting algorithm for surface reconstruction.” IEEE Transactions on
    Visualization and Computer Graphics 5 (1999): 349-359.

    Function that computes a triangle mesh from a oriented PointCloud. This implements the Ball Pivoting algorithm
    proposed in F. Bernardini et al., “The ball-pivoting algorithm for surface reconstruction”, 1999.
    The implementation is also based on the algorithms outlined in Digne, “An Analysis and Implementation of a
    Parallel Ball Pivoting Algorithm”, 2014. The surface reconstruction is done by rolling a ball with a given
    radius over the point cloud, whenever the ball touches three points a triangle is created.

    Radius is computed to be slightly larger than the average distance between points
    https://cs184team.github.io/cs184-final/writeup.html

    Parameters
    ----------
    pcd: PointCloud
        Point cloud object
    radii: Union[list, float, None], default=0.6
        Radius (unique or a list) of the ball.

    Returns
    -------
    mesh: Mesh
        Mesh object
    """
    # Convert the point cloud with normals to an open3d format

    # init o3d point cloud
    if pcd.o3d_pcd is None:
        pcd.set_o3d_pcd_from_df()

    # # add normals
    # if not pcd.has_normals:
    #     raise ValueError("Some normal components are not included in the df point cloud (either 'n_x', or 'n_y', "
    #                      "or 'n_z'). Please launch again the normal computation.")
    # else:
    #     pcd.o3d_pcd.normals = o3d.utility.Vector3dVector(pcd.df[["n_x", "n_y", "n_z"]].to_numpy())

    # Mesh point cloud
    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd.o3d_pcd, o3d.utility.DoubleVector(radii)
    )

    # Init Mesh object
    mesh = Mesh(pcd=pcd.df, o3d_pcd=pcd.o3d_pcd, o3d_mesh=o3d_mesh)
    mesh.set_df_from_o3d_mesh()

    return mesh


def poisson_reconstruction(
    pcd: PointCloud,
    depth: int = 8,
    width: int = 0,
    scale: float = 1.1,
    linear_fit: bool = False,
    n_threads: int = -1,
) -> Mesh:
    """
    Kazhdan, Michael M. et al. “Poisson surface reconstruction.” SGP '06 (2006).

    Function that computes a triangle mesh from a oriented PointCloud pcd. This implements the Screened Poisson
    Reconstruction proposed in Kazhdan and Hoppe, “Screened Poisson Surface Reconstruction”, 2013. This function
    uses the original implementation by Kazhdan.
    See https://github.com/mkazhdan/PoissonRecon

    Warning: Since the Poisson reconstruction is an optimisation process, the triangle vertices are not the initial
    point cloud points. Thus, "pcd" points will be different from "o3d..." and mesh points.

    Parameters
    ----------
    pcd: PointCloud
        Point cloud object
    depth: int, default=8
        Maximum depth of the tree that will be used for surface reconstruction. Running at depth d corresponds to
        solving on a grid whose resolution is no larger than 2^d x 2^d x 2^d. Note that since the reconstructor
        adapts the octree to the sampling density, the specified reconstruction depth is only an upper bound.
    width: int, default=0
        Specifies the target width of the finest level octree cells. This parameter is ignored if depth is specified
    scale: float, default=1.1
        Specifies the ratio between the diameter of the cube used for reconstruction and the diameter of the
        samples’ bounding cube.
    linear_fit: bool, default=False
        If true, the reconstructor will use linear interpolation to estimate the positions of iso-vertices.
    n_threads: int, default=-1
        Number of threads used for reconstruction. Set to -1 to automatically determine it.

    Returns
    -------
    mesh: Mesh
        Mesh object
    """
    # Convert the point cloud with normals to an open3d format

    # init o3d point cloud
    o3d_pcd = df2o3d(pcd.df)

    # # add normals
    # if not pcd.has_normals:
    #     raise ValueError("Some normal components are not included in the df point cloud (either 'n_x', or 'n_y', "
    #                      "or 'n_z'). Please launch again the normal computation.")
    # else:
    #     o3d_pcd.normals = o3d.utility.Vector3dVector(pcd.df[["n_x", "n_y", "n_z"]].to_numpy())

    # Mesh point cloud
    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        o3d_pcd, depth, width, scale, linear_fit, n_threads
    )

    # # Init Mesh object
    # mesh = Mesh(pcd=pcd.df, o3d_pcd=o3d.geometry.PointCloud(points=o3d_mesh[0].vertices), o3d_mesh=o3d_mesh[0])
    # mesh.set_df_from_vertices(np.asarray(o3d_mesh[0].triangles))

    # Init Mesh object
    # TODO: Check consistency between pcd.df list of points and o3d list of points
    mesh = Mesh(
        pcd=pcd.df,
        o3d_pcd=o3d.geometry.PointCloud(points=o3d_mesh[0].vertices),
        o3d_mesh=o3d_mesh[0],
    )
    mesh.set_df_from_o3d_mesh()

    return mesh


def delaunay_2d_reconstruction(
    pcd: PointCloud, method: str = "matplotlib"
) -> Mesh:
    """
    2.5D Delaunay triangulation: Delaunay triangulation on the planimetric points and add afterwards the z coordinates.

    Parameters
    ----------
    pcd: PointCloud
        Point cloud object
    method: str, default='matplotlib'
        Method to use for Delaunay 2.5D triangulation. Available methods are 'matplotlib' and 'scipy'.

    Returns
    -------
    mesh: Mesh
        Mesh object
    """

    # Delaunay triangulation

    mesh = Mesh(pcd=pcd.df)

    if method == "scipy":
        mesh_data = Delaunay(pcd.df[["x", "y"]].to_numpy())
        mesh.set_df_from_vertices(mesh_data.simplices)
        return mesh

    elif method == "matplotlib":
        mesh_data = mtri.Triangulation(pcd.df.x, pcd.df.y)
        mesh.set_df_from_vertices(mesh_data.triangles)
        return mesh

    else:
        raise NotImplementedError(
            f"Unknown library for Delaunay triangulation: '{method}'. Should either be 'scipy' "
            f"or 'matplotlib'."
        )
