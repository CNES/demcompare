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
Parameters for cars-mesh tool
"""

from .core.denoise_pcd import bilateral_filtering
from .core.filter import (
    local_density_analysis,
    radius_filtering_outliers_o3,
    statistical_filtering_outliers_o3d,
)
from .core.mesh import (
    ball_pivoting_reconstruction,
    delaunay_2d_reconstruction,
    poisson_reconstruction,
)
from .core.simplify_mesh import (
    simplify_quadric_decimation,
    simplify_vertex_clustering,
)
from .core.texture import texturing

TRANSITIONS_METHODS = {
    "filter": {
        "radius_o3d": radius_filtering_outliers_o3,
        "statistics_o3d": statistical_filtering_outliers_o3d,
        "local_density_analysis": local_density_analysis,
    },
    "denoise_pcd": {"bilateral": bilateral_filtering},
    "mesh": {
        "delaunay_2d": delaunay_2d_reconstruction,
        "poisson": poisson_reconstruction,
        "bpa": ball_pivoting_reconstruction,
    },
    "simplify_mesh": {
        "garland-heckbert": simplify_quadric_decimation,
        "vertex_clustering": simplify_vertex_clustering,
    },
    "denoise_mesh": {},
    "texture": {"texturing": texturing},
}

PCD_FILE_EXTENSIONS = ["ply", "las", "laz"]

MESH_FILE_EXTENSIONS = ["ply"]

INITIAL_STATES = ["initial_pcd", "meshed_pcd"]
