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

import pandas as pd
import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.tri as mtri

from tools.point_cloud_handling import df2o3d


def ball_pivoting_reconstruction(df_pcd: pd.DataFrame, radii: Union[list, float, None] = 0.6) -> dict:
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
    df_pcd: pd.DataFrame
        Point cloud with the extra information attached by point
    radii: Union[list, float, None], default=0.6
        Radius (unique or a list) of the ball.

    Returns
    -------
    :dict
       Dictionary with four keys:
            - 'pcd': Point cloud and extra information expressed in a pandas DataFrame (=df_pcd)
            - 'mesh': numpy array of the triangle vertex indexes regarding the df_pcd
            - 'o3d_pcd': point cloud in open3d format
            - 'o3d_mesh' mesh in open3d format
    """
    # Convert the point cloud with normals to an open3d format

    # init o3d point cloud
    o3d_pcd = df2o3d(df_pcd)

    # add normals
    if ("n_x" not in df_pcd) or ("n_y" not in df_pcd) or ("n_z" not in df_pcd):
        raise ValueError("Some normal components are not included in the df point cloud (either 'n_x', or 'n_y', "
                         "or 'n_z'). Please launch again the normal computation.")
    else:
        o3d_pcd.normals = o3d.utility.Vector3dVector(df_pcd[["n_x", "n_y", "n_z"]].to_numpy())

    # Mesh point cloud
    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(o3d_pcd, o3d.utility.DoubleVector(radii))

    return {"pcd": df_pcd, "mesh": np.asarray(o3d_mesh.triangles), "o3d_pcd": o3d_pcd, "o3d_mesh": o3d_mesh}


def poisson_reconstruction(df_pcd: pd.DataFrame,
                           depth: int = 8,
                           width: int = 0,
                           scale: float = 1.1,
                           linear_fit: bool = False,
                           n_threads: int = -1) -> dict:
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
    df_pcd: pd.DataFrame
        Point cloud with the extra information attached by point
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
    :dict
       Dictionary with four keys:
            - 'pcd': Point cloud and extra information expressed in a pandas DataFrame (=df_pcd)
            - 'mesh': numpy array of the triangle vertex indexes. Warning: the points concerned are the ones in
            'o3d_pcd'
            - 'o3d_pcd': point cloud (vertices of the mesh) in open3d format
            - 'o3d_mesh' mesh in open3d format
    """
    # Convert the point cloud with normals to an open3d format

    # init o3d point cloud
    o3d_pcd = df2o3d(df_pcd)

    # add normals
    if ("n_x" not in df_pcd) or ("n_y" not in df_pcd) or ("n_z" not in df_pcd):
        raise ValueError("Some normal components are not included in the df point cloud (either 'n_x', or 'n_y', "
                         "or 'n_z'). Please launch again the normal computation.")
    else:
        o3d_pcd.normals = o3d.utility.Vector3dVector(df_pcd[["n_x", "n_y", "n_z"]].to_numpy())

    # Mesh point cloud
    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pcd,
                                                                         depth,
                                                                         width,
                                                                         scale,
                                                                         linear_fit,
                                                                         n_threads)

    return {"pcd": df_pcd, "mesh": np.asarray(o3d_mesh[0].triangles), "o3d_pcd": o3d.geometry.PointCloud(points=o3d_mesh[0].vertices), "o3d_mesh": o3d_mesh[0]}


def delaunay_2d_reconstruction(df_pcd: pd.DataFrame, method: str = "matplotlib") -> dict:
    """
    2.5D Delaunay triangulation: Delaunay triangulation on the planimetric points and add afterwards the z coordinates.

    Parameters
    ----------
    df_pcd: pd.DataFrame
        Point cloud with the extra information attached by point
    method: str, default='matplotlib'
        Method to use for Delaunay 2.5D triangulation. Available methods are 'matplotlib' and 'scipy'.

    Returns
    -------
    :dict
       Dictionary with two keys:
            - 'pcd': Point cloud and extra information expressed in a pandas DataFrame (=df_pcd)
            - 'mesh': numpy array of the triangle vertex indexes regarding the df_pcd
    """

    # Delaunay triangulation

    if method == "scipy":
        mesh = Delaunay(df_pcd[["x", "y"]].to_numpy())
        return {"pcd": df_pcd, "mesh": mesh.simplices}

    elif method == "matplotlib":
        mesh = mtri.Triangulation(df_pcd.x, df_pcd.y)
        return {"pcd": df_pcd, "mesh": mesh.triangles}

    else:
        raise NotImplementedError(f"Unknown library for Delaunay triangulation: '{method}'. Should either be 'scipy' "
                                  f"or 'matplotlib'.")
