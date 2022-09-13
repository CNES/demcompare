<div align="center">
  <a href="https://gitlab.cnes.fr/cars/mesh_3d"><img src="docs/source/images/picto_transparent.png" alt="CARS" title="CARS"  width="20%"></a>

<h4>Mesh 3D</h4>

[![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)

<p>
  <a href="#overview">Overview</a> •
  <a href="#requirements">Requirements</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contribution">Contribution</a> •
  <a href="#references">References</a>
</p>
</div>

## Overview

Mesh 3D short description:

"3D Surface reconstruction with texture and classification from remote sensing photogrammetric point cloud"


* Free software: Apache Software License 2.0

[//]: # (* Documentation: https://mesh-3d.readthedocs.io.)


## Requirements

    importlib           ; python_version>"3.8"
    argparse                      # Python Argument Parser
    argcomplete                   # Autocompletion Argparse
    numpy                         # array manipulation
    laspy                         # las file manipulation
    open3d                        # 3D library open source
    pandas                        # data with many attributes manipulation
    scipy                         # scientific library
    plyfile                       # ply file manipulation
    matplotlib                    # visualisation and meshing
    loguru                        # logs handler
    pyproj                        # coordinates conversion
    transitions                   # state machine
    rasterio                      # georeferenced data handler

## Features

TODO

- Feature 1
- Feature 2
- ...

## Quick Start

### Installation

Git clone the repository, open a terminal and launch the following commands:
```bash
cd path/to/dir/mesh_3d
make install
```

### Execute

You can run too functions with the `mesh_3d` cli:
* `mesh_3d reconstruct` launches the 3D reconstruction pipeline according to the user specifications
* `mesh_3d evaluate` computes metrics between two point clouds and saves visuals for qualitative analysis

#### Reconstruct

Configure the pipeline in a JSON file `/path/to/config.json`:
```json
{
  "input_path": "/path/to/input/data.ply",
  "output_dir": "/path/to/output_dir",
  "initial_state": "initial_pcd",
  "tif_img_path": "/path/to/tif_img_texture.tif",
  "rpc_path": "/path/to/rpc.XML",
  "utm_code": 32631,
  "state_machine": [
    {
      "action": "filter",
      "method": "radius_o3d",
      "params": {
        "radius":  3
      }
    },
    {
      "action": "denoise_pcd",
      "method": "bilateral_filtering",
      "params": {
        "use_open3d":  true
      }
    },
    {
      "action": "mesh",
      "method": "delaunay_2d",
      "params": {}
    },
    {
      "action": "simplify_mesh",
      "method": "garland-heckbert",
      "params": {
        "reduction_ratio_of_triangles": 0.75
      }
    },
    {
      "action":  "texture",
      "method": "texturing",
      "params": {}
    }
  ]
}
```

Where:
* `input_path`: Filepath to the input. Should either be a point cloud or a mesh.
* `output_dir`: Directory path to the output folder where to save results.
* `initial_state` (optional, default=`"initial_pcd"`): Initial state in the state machine. If you input a point cloud, 
it should be `"initial_pcd"`. If you input a mesh, it could either be `"initial_pcd"` (you can compute new 
values over the points) or `"meshed_pcd"` (if for instance you only want to texture an already existing mesh).
* `state_machine`: List of steps to process the input according to a predefined state machine (see below). 
Each step has three possible keys:`action` (str) which corresponds to the trigger name, `method` (str) which 
specifies the method to use to do that step (possible methods are available in the `/mesh_3d/param.py` file,
by default it is the first method that is selected), `params` (dict) which specifies in a dictionary the parameters 
for each method.
<img src="fig_state_machine.png">

If a texturing step is specified, then the following parameters become mandatory:
* `rpc_path`: Path to the RPC xml file
* `tif_img_path`: Path to the TIF image from which to extract the texture image
* `utm_code`: The UTM code of the point cloud coordinates expressed as a EPSG code number for transformation purpose


Finally, you can launch the following commands to activate the virtual environment and run the pipeline:
```bash
source /venv/bin/activate
mesh_3d reconstruct /path/to/config.json
```

#### Evaluate

The evaluation function computes a range of metrics between two point clouds and outputs visuals for 
qualitative analysis.

Configure the pipeline in a JSON file `/path/to/config.json`:
```json
{
  "input_path_1": "/path/to/point_cloud/or/mesh_1.ply",
  "input_path_2": "/path/to/point_cloud/or/mesh_2.ply",
  "output_dir": "/path/to/output_dir"
}
```

Where:
* `input_path_1`: Filepath to the first input. Should either be a point cloud or a mesh.
* `input_path_2`: Filepath to the second input. Should either be a point cloud or a mesh.
* `output_dir`: Directory path to the output folder where to save results.

Finally, you can launch the following commands to activate the virtual environment and run the evaluation:
```bash
source /venv/bin/activate
mesh_3d evaluate /path/to/config.json
```

## Documentation


Go in docs/ directory



* Documentation: https://mesh-3d.readthedocs.io.


## Contribution

See [Contribution](CONTRIBUTING.md) manual


* Free software: Apache Software License 2.0


## References

This package was created with cars-cookiecutter project template.

Inspired by [main cookiecutter template](https://github.com/audreyfeldroy/cookiecutter-pypackage) and 
[AI4GEO cookiecutter template](https://gitlab.cnes.fr/ai4geo/lot2/cookiecutter-python)