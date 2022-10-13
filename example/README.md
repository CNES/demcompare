# Example

This folder provides small data to test the pipeline extracted from Toulouse Pleiades triplet acquired on 27/09/2013 and contains:
* `point_cloud.laz`: a point cloud generated with CARS with panchromatic images P01 and P02,
* `rpc.XML`: a RPC file of panchromatic P01,
* `texture_image.tif`: a texture image extracted from panchromatic P01 (top left point at (col, row) = (15029, 17016)),
* `config.json`: an example of configuration to launch the reconstruction pipeline.

To launch the code, please follow the guidelines below:
```bash
# Install Mesh3D library
cd path/to/dir/mesh_3d
make install

# Activate virtual environment
source venv/bin/activate

# Launch reconstruction
mesh3d reconstruct example/config.json
```