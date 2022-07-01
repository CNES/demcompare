# TRANSITIONS_METHODS = {
#     "filter": ["radius", "statistics"],
#     "denoise_pcd": ["bilateral"],
#     "mesh": ["delaunay_2d"],
#     "denoise_mesh": ["bilateral"],
#     "texture": ["main"]
# }

from .core.filter import radius_filtering_outliers_o3, statistical_filtering_outliers_o3d
from .core.denoise import bilateral_denoising
from .core.mesh import delaunay_2d_reconstruction, poisson_reconstruction, ball_pivoting_reconstruction


TRANSITIONS_METHODS = {
    "filter": {
        "radius_o3d": radius_filtering_outliers_o3,
        "statistics_o3d": statistical_filtering_outliers_o3d
    },
    "denoise_pcd": {
        "bilateral": bilateral_denoising
    },
    "mesh": {
        "delaunay_2d": delaunay_2d_reconstruction,
        "poisson": poisson_reconstruction,
        "bpa": ball_pivoting_reconstruction
    },
    "denoise_mesh": {},
    "texture": {}
}

PCD_FILE_EXTENSIONS = ["ply", "las", "laz"]

MESH_FILE_EXTENSIONS = ["ply"]

INITIAL_STATES = ["initial_pcd", "meshed_pcd"]
