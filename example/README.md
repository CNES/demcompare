# Example

This folder provides small data to test the pipeline extracted from Toulouse Pleiades triplet acquired on 27/09/2013 and contains:

* `point_cloud.laz`: a point cloud generated with CARS with panchromatic images P01 and P02,
* `rpc.XML`: a RPC file of panchromatic P01,
* `texture_image.tif`: a texture image extracted from panchromatic P01 (top left point at (col, row) = (15029, 17016)),
* `config_reconstruct.json`: an example of configuration to run the reconstruction pipeline.
* `config_evaluate.json`: an example of configuration to run the evaluation pipeline between the input point cloud and the output of the reconstruction pipeline.

To run the code, please follow the guidelines below:

```bash
# Install CARS-MESH library
cd path/to/dir/cars-mesh
make install

# Activate virtual environment
source venv/bin/activate

# Go to example directory
cd example/

# Run reconstruction
cars-mesh config_reconstruct.json

# Run evaluation between the input point cloud and the vertices of the reconstructed mesh
cars-mesh-evaluate config_evaluate.json
```

For more information, please check the [cars-mesh documentation](https://github.com/cnes/cars-mesh).
