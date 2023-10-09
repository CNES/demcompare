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
Simplifying methods for the mesh to decrease the number of faces
"""

# Standard imports
from typing import Union

# Third party imports
import numpy as np
import open3d as o3d

# Cars-mesh imports
from ..tools.handlers import Mesh


def simplify_quadric_decimation(
    mesh: Mesh,
    reduction_ratio_of_triangles: float = 0.9,
    target_number_of_triangles: Union[int, None] = None,
    maximum_error: float = float("inf"),
    boundary_weight: float = 1.0,
) -> Mesh:
    """
    Function to simplify mesh using Quadric Error Metric Decimation by
    Garland and Heckbert

    Parameters
    ----------
    mesh: Mesh
        Mesh object
    reduction_ratio_of_triangles: float (default=0.9)
        Reduction ratio of triangles (for instance, 0.9 to keep 90% of the
        triangles)
    target_number_of_triangles: int or None (default=None)
        The number of triangles that the simplified mesh should have. It is
        not guaranteed that this number will be reached. If both
        reduction_ratio_of_triangles and target_number_of_triangles are
        specified, the latter will be used.
    maximum_error: float (default=inf)
        The maximum error where a vertex is allowed to be merged
    boundary_weight: float (default=1.0)
        A weight applied to edge vertices used to preserve boundaries

    Returns
    -------
    out_mesh: Mesh
        Simplified mesh object
    """

    # Check that mesh has triangle faces
    if not mesh.has_triangles:
        raise ValueError(
            "Mesh has no triangle faces. Please apply a mesh reconstruction "
            "algorithm before."
        )

    # Create an open3d mesh if not present
    if mesh.o3d_mesh is None:
        mesh.set_o3d_mesh_from_df()

    # Check and set parameters
    if target_number_of_triangles is not None:
        target_number_of_triangles = int(target_number_of_triangles)

    elif reduction_ratio_of_triangles is not None:
        reduction_ratio_of_triangles = float(reduction_ratio_of_triangles)
        if (
            reduction_ratio_of_triangles > 1.0
            or reduction_ratio_of_triangles < 0.0
        ):
            raise ValueError(
                f"'reduction_ratio_of_triangles' should be contained in "
                f"[0, 1]. Here found: "
                f"{reduction_ratio_of_triangles}."
            )

        target_number_of_triangles = int(
            np.asarray(mesh.o3d_mesh.triangles).shape[0]
            * reduction_ratio_of_triangles
        )

    else:
        raise ValueError(
            f"Either 'reduction_ratio_of_triangles' or 'target_number_of_"
            f"triangles' should be specified. Here found:\n"
            f"- 'reduction_ratio_of_triangles' = "
            f"{reduction_ratio_of_triangles}\n"
            f"- 'target_number_of_triangles' = {target_number_of_triangles}"
        )

    # Simplify
    out_mesh_o3d = mesh.o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(target_number_of_triangles),
        maximum_error=float(maximum_error),
        boundary_weight=float(boundary_weight),
    )

    # Create a new mesh object
    out_mesh = Mesh(o3d_mesh=out_mesh_o3d)
    out_mesh.set_df_from_o3d_mesh()

    return out_mesh


def simplify_vertex_clustering(
    mesh: Mesh,
    dividing_size: float = 16.0,
    contraction=o3d.geometry.SimplificationContraction.Average,
) -> Mesh:
    """
    Function to simplify mesh using vertex clustering.

    Parameters
    ----------
    mesh: Mesh
        Mesh object
    dividing_size: float
        A new voxel size is computed as
        max(max_bound - min_bound) / dividing size
    contraction: open3d.geometry.SimplificationContraction
    (default=<SimplificationContraction.Average – 0>)
        Method to aggregate vertex information. Average computes a simple
        average, Quadric minimizes the distance to the adjacent planes.

    Returns
    -------
    out_mesh: Mesh
        Simplified mesh object
    """

    # Check that mesh has triangle faces
    if not mesh.has_triangles:
        raise ValueError(
            "Mesh has no triangle faces. Please apply a mesh reconstruction "
            "algorithm before."
        )

    # Create an open3d mesh if not present
    if mesh.o3d_mesh is None:
        mesh.set_o3d_mesh_from_df()

    # Simplify
    dividing_size = float(dividing_size)
    if dividing_size == 0.0:
        raise ValueError("Dividing size needs to be > 0.")

    voxel_size = (
        max(
            mesh.o3d_mesh.get_max_bound() - mesh.o3d_mesh.get_min_bound()
        ).item()
        / dividing_size
    )

    out_mesh_o3d = mesh.o3d_mesh.simplify_vertex_clustering(
        voxel_size=float(voxel_size), contraction=contraction
    )

    # Create a new mesh object
    out_mesh = Mesh(o3d_mesh=out_mesh_o3d)
    out_mesh.set_df_from_o3d_mesh()

    return mesh
