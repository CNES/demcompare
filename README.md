<div align="center">
<a target="_blank" href="https://github.com/CNES/cars-mesh">
<picture>
  <source
    srcset="https://raw.githubusercontent.com/CNES/cars-mesh/master/docs/source/images/picto_dark.png"
    media="(prefers-color-scheme: dark)"
  />
  <img
    src="https://raw.githubusercontent.com/CNES/cars-mesh/master/docs/source/images/picto_light.png"
    alt="CARS"
    width="40%"
  />
</picture>
</a>

<h4>CARS-MESH</h4>


[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0/)
[![Documentation](https://readthedocs.org/projects/cars-mesh/badge/?version=stable)](https://cars-mesh.readthedocs.io/?badge=stable)



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

CARS-MESH is a library to do 3D Surface reconstruction with texture and classification from remote sensing photogrammetric point cloud.

This tool is part of 3D CNES tools around [CARS](https://github.com/cnes/cars) but can be run independently. CARS can be used to generate input point clouds from stereo images.

CARS-MESH depends heavily on [open3D](http://www.open3d.org/) toolbox.

CARS-MESH is a study project and is not recommended for production. The project is for experimental use only, with no guaranty on stability.

* Free software: Apache Software License 2.0

[//]: # (* Documentation: https://cars-mesh.readthedocs.io.)

## Features

CARS-MESH outputs a textured 3D mesh from a point cloud. The main steps currently implemented
are respectively:

* point cloud outlier filtering
* point cloud denoising
* meshing
* mesh denoising *(note: no method is implemented for now)*
* mesh simplification
* texturing

It can be run on a point cloud or directly on a mesh if only the last steps are chosen.

In the meantime, CARS-MESH provides a simple point clouds comparison and evaluation tool.
It computes a bunch of metrics between two point clouds or meshes (vertices are used
for the comparison for mesh) and gives a visual glimpse of the local distances from one point cloud to the other.

## Install

### From Pypi

Create a Python virtual environment, git clone the repository and install the library.

```bash
# Create your virtual environment "venv"
python -m venv venv

# Activate your venv (on UNIX)
source venv/bin/activate

# Update pip and setuptools package
python -m pip --upgrade pip setuptools

# Install the cars-mesh tool
python -m pip install cars-mesh

# Test if it works
cars-mesh -h
```

### From source

Git clone the repository, open a terminal and run the following commands:

```bash

# Clone repository
git clone https://github.com/CNES/cars-mesh.git

# Install
make install

# Activate your venv (on UNIX)
source venv/bin/activate

# Test if it works
cars-mesh -h
```

## CARS-MESH usage

`cars-mesh` command runs a 3D reconstruction pipeline according to the user specifications

Configure the pipeline in a JSON file `/path/to/config_reconstruct.json`:

```json
{
  "input_path": "point_cloud.laz",
  "output_dir": "output_reconstruct",
  "output_name": "textured_mesh",
  "rpc_path": "rpc.XML",
  "tif_img_path": "texture_image.tif",
  "image_offset": [
    15029,
    17016
  ],
  "utm_code": 32631,
  "state_machine": [
    {
      "action": "filter",
      "method": "radius_o3d",
      "save_output": true,
      "params": {
        "radius": 4,
        "nb_points": 25
      }
    },
    {
      "action": "denoise_pcd",
      "method": "bilateral",
      "save_output": true,
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
      "save_output": true,
      "params": {
        "method": "scipy"
      }
    },
    {
      "action": "simplify_mesh",
      "method": "garland-heckbert",
      "save_output": true,
      "params": {
        "reduction_ratio_of_triangles": 0.75
      }
    },
    {
      "action": "texture",
      "method": "texturing",
      "params": {}
    }
  ]
}
```

Where:

* `input_path`: Filepath to the input. Should either be a point cloud or a mesh.
* `output_dir`: Directory path to the output folder where to save results.
* `output_name` (optional, default=`output_cars-mesh`): Name of the output mesh file (without extension)
* `initial_state` (optional, default=`"initial_pcd"`): Initial state in the state machine. If you input a point cloud,
it should be `"initial_pcd"`. If you input a mesh, it could either be `"initial_pcd"` (you can compute new
values over the points) or `"meshed_pcd"` (if for instance you only want to texture an already existing mesh).
* `state_machine`: List of steps to process the input according to a predefined state machine (see below).
Each step has three possible keys:`action` (str) which corresponds to the trigger name, `method` (str) which
specifies the method to use to do that step (possible methods are available in the `/cars_mesh/param.py` file,
by default it is the first method that is selected), `params` (dict) which specifies in a dictionary the parameters
for each method.
<img src="fig_state_machine.png">

For each step, you can specify whether to save the intermediate output to disk.
To do so, in the step dictionary, you need to specify a key `save_output` as `true` (by default, it is `false`).
It will create a folder in the output directory named "intermediate_results" where these intermediate results will be saved.

If a texturing step is specified, then the following parameters become mandatory:

* `rpc_path`: Path to the RPC xml file
* `tif_img_path`: Path to the TIF image from which to extract the texture image. For now, it should be the whole satellite image to be consistent with the product's RPC.
* `utm_code`: The UTM code of the point cloud coordinates expressed as a EPSG code number for transformation purpose. *Warning: the input cloud is assumed to be expressed in a UTM frame.*

Another parameter - optional - when applying a texture is the `image_offset`.
It is possible to use a cropped version of the image texture as long as the `image_offset` parameter is specified.
It is a tuple or a list of two elements (col, row) corresponding to the top left corner coordinates of the cropped image texture.
It will change the normalisation offset of the RPC data to make the texture fit to the point cloud.
If the image is only cropped on the bottom right side of the image, no offset information is needed.

Finally, you can run the following commands to activate the virtual environment and run the pipeline:

```bash
source venv/bin/activate
cars-mesh config_reconstruct.json
```

## CARS-MESH Evaluate

`cars-mesh-evaluate` tool computes metrics between two point clouds and saves visuals for qualitative analysis (If an input is a mesh, its vertices will be used for comparison)

Configure the evaluate pipeline in a JSON file `config_evaluate.json`:

```json
{
  "input_path_1": "example/point_cloud.laz",
  "input_path_2": "example/output/textured_mesh.ply",
  "output_dir": "example/output_evaluate"
}
```

Where:

* `input_path_1`: Filepath to the first input. Should either be a point cloud or a mesh.
* `input_path_2`: Filepath to the second input. Should either be a point cloud or a mesh.
* `output_dir`: Directory path to the output folder where to save results.

Finally, you can run the following commands to activate the virtual environment and run the evaluation:

```bash
source venv/bin/activate
cars-mesh-evaluate /path/to/config_evaluate.json
```

*N.B.: To run the example above, you need to run the example reconstruction pipeline first (cf previous section)*

## Example

Please find data example to run the pipeline in [here](example) and guidelines [over here](example/README.md).

## Documentation

Run the following commands to build the doc:

```bash
source venv/bin/activate
make docs
```

The Sphinx documentation should pop in a new tab of your browser.

[//]: # (Documentation: https://cars-mesh.readthedocs.io)

## Tests

Run the following commands to run the tests:

```bash
source /venv/bin/activate
make test
```

*Warning: there are no tests on Poisson reconstruction. (cf the "Perspectives" section on mesh, and the documentation Core/Mesh/PoissonReconstruction).*

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
