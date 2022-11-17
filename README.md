<div align="center">
  <a href="https://gitlab.cnes.fr/cars/mesh3d"><img src="docs/source/images/picto_transparent.png" alt="CARS" title="CARS"  width="20%"></a>

<h4>Mesh 3D</h4>

[![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0/)
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

Mesh3D is a library allow to do 3D Surface reconstruction with texture and classification from remote sensing photogrammetric point cloud.


* Free software: Apache Software License 2.0

[//]: # (* Documentation: https://mesh3d.readthedocs.io.)


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

Mesh3D allows to output a textured 3D mesh from a point cloud. The main steps currently implemented
are respectively:
* point cloud outlier filtering
* point cloud denoising
* meshing
* mesh denoising *(note: no method is implemented for now)*
* mesh simplification
* texturing

It can be run on a point cloud or directly on a mesh if only the last steps are to be passed.

In the meantime, Mesh3D provides a point clouds comparison and evaluation tool. 
It computes a bunch of metrics between two point clouds and gives a visual glimpse of the local distances from one point cloud to the other.

## Quick Start

### Installation

Git clone the repository, open a terminal and launch the following commands:
```bash
cd path/to/dir/mesh3d
make install
```

### Execute

You can run two functions with the `mesh3d` cli:
* `mesh3d reconstruct` launches the 3D reconstruction pipeline according to the user specifications
* `mesh3d evaluate` computes metrics between two point clouds and saves visuals for qualitative analysis

#### Reconstruct

Configure the pipeline in a JSON file `/path/to/config.json`:
```json
{
  "input_path": "/path/to/input/data.las",
  "output_dir": "/path/to/output_dir",
  "initial_state": "initial_pcd",
  "tif_img_path": "/path/to/tif_img_texture.tif",
  "rpc_path": "/path/to/rpc.XML",
  "image_offset": null,
  "utm_code": 32631,
  "state_machine": [
    {
      "action": "filter",
      "method": "radius_o3d",
      "params": {
        "radius":  2,
        "nb_points": 12
      }
    },
    {
      "action": "denoise_pcd",
      "method": "bilateral",
      "params": {
        "num_iterations": 10,
        "neighbour_kdtree_dict": {
          "knn": 10,
          "num_workers_kdtree": 6
        },
        "neighbour_normals_dict": {
          "knn_normals": 10,
          "use_open3d": true
        },
        "sigma_d": 1.5,
        "sigma_n": 1,
        "num_chunks": 2
      }
    },
    {
      "action": "mesh",
      "method": "delaunay_2d",
      "params": {
        "method": "scipy"
      }
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
specifies the method to use to do that step (possible methods are available in the `/mesh3d/param.py` file,
by default it is the first method that is selected), `params` (dict) which specifies in a dictionary the parameters
for each method.
<img src="fig_state_machine.png">

If a texturing step is specified, then the following parameters become mandatory:
* `rpc_path`: Path to the RPC xml file
* `tif_img_path`: Path to the TIF image from which to extract the texture image. For now, it should be the whole satellite image to be consistent with the product's RPC.
* `utm_code`: The UTM code of the point cloud coordinates expressed as a EPSG code number for transformation purpose. *Warning: the input cloud is assumed to be expressed in a UTM frame.*

Another parameter - optional - when applying a texture is the `image_offset`.
It is possible to use a cropped version of the image texture as long as the `image_offset` parameter is specified.
It is a tuple or a list of two elements (col, row) corresponding to the top left corner coordinates of the cropped image texture.
It will change the normalisation offset of the RPC data to make the texture fit to the point cloud.
If the image is only cropped on the bottom right side of the image, no offset information is needed.

Finally, you can launch the following commands to activate the virtual environment and run the pipeline:
```bash
source /venv/bin/activate
mesh3d reconstruct /path/to/config.json
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
source venv/bin/activate
mesh3d evaluate /path/to/config.json
```

## Example

Please find data example to launch the pipeline in [here](example) and guidelines [over here](example/README.md).


## Documentation

Run the following commands to build the doc:
```bash
source /venv/bin/activate
make docs
```
The Sphinx documentation should pop in a new tab of your browser.


[//]: # (Documentation: https://mesh-3d.readthedocs.io)

## Tests
Run the following commands to run the tests:
```bash
source /venv/bin/activate
make test
```
*Warning: there are no tests on Poisson reconstruction. (cf the TO DO part on mesh, and the documentation Core/Mesh/PoissonReconstruction).*

## Perspectives

* **General**
  * [ ] Add the possibility to use semantic maps and modify functions to take them into account for processing (for example building roofs could be processed differently from roads).
  * [ ] Recover correlation metrics from previous CARS processing and add it as an input to exploit them in further processings.
  * [ ] Make sure information in the PointCloud pandas DataFrame object are the same as the ones in the Point Cloud open3d object all along the process.
  * [ ] To make it more large scale with potentially large point clouds, las files should be read by chunk (cf [LASPY documentation](https://laspy.readthedocs.io/en/latest/basic.html#chunked-writing))


* **Filtering of outliers**
  * [ ] Integrate the use of CARS already existing functions (in its latest version)

* **Mesh** 
  * [ ] Texturing step can fail after a Poisson reconstruction because of the outliers created by this method:
    * [ ] Adapt the parameters of the method such as width 
    * [ ] Clean the point cloud after Poisson mesh to remove those blocking outliers
  * [ ] Add to the tests Poisson reconstruction

* **Texturing**
  * [ ] Make it satellite agnostic (for now it takes into account Pleiades imagery)
  * [ ] Handle multiple texture images
  * [ ] Handle occlusions
  * [ ] Make percentiles (for better texture visualisation) computation large scale (avoid having to load the full raster in memory). It can be done by computing percentiles only on a random portion of pixels (like 20%)


## Contribution

See [Contribution](CONTRIBUTING.md) manual


* Free software: Apache Software License 2.0


## References

This package was created with cars-cookiecutter project template.

Inspired by [main cookiecutter template](https://github.com/audreyfeldroy/cookiecutter-pypackage) and 
[AI4GEO cookiecutter template](https://gitlab.cnes.fr/ai4geo/lot2/cookiecutter-python)
