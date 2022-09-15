from .core.denoise_pcd import bilateral_filtering
from .core.filter import (
    radius_filtering_outliers_o3,
    statistical_filtering_outliers_o3d,
    local_density_analysis
)
from .core.mesh import (
    ball_pivoting_reconstruction,
    delaunay_2d_reconstruction,
    poisson_reconstruction,
)
from .core.simplify_mesh import (
    simplify_quadric_decimation,
    simplify_vertex_clustering
)
from .core.texture import texturing

TRANSITIONS_METHODS = {
    "filter": {
        "radius_o3d": radius_filtering_outliers_o3,
        "statistics_o3d": statistical_filtering_outliers_o3d,
        "local_density_analysis": local_density_analysis
    },
    "denoise_pcd": {"bilateral": bilateral_filtering},
    "mesh": {
        "delaunay_2d": delaunay_2d_reconstruction,
        "poisson": poisson_reconstruction,
        "bpa": ball_pivoting_reconstruction,
    },
    "simplify_mesh": {
        "garland-heckbert": simplify_quadric_decimation,
        "vertex_clustering": simplify_vertex_clustering
    },
    "denoise_mesh": {},
    "texture": {"texturing": texturing},
}

PCD_FILE_EXTENSIONS = ["ply", "las", "laz"]

MESH_FILE_EXTENSIONS = ["ply"]

INITIAL_STATES = ["initial_pcd", "meshed_pcd"]
