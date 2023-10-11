
.. _usage_cars-mesh:

===============
Use CARS-MESH 
===============

``cars-mesh`` runs the 3D reconstruction pipeline according to the user specifications

To run the 3D reconstruction pipeline, you first need to configure it in a JSON file like the following:

.. code-block:: JSON

    {
      "input_path": "example/point_cloud.laz",
      "output_dir": "example/output_reconstruct",
      "output_name": "textured_mesh",
      "rpc_path": "example/rpc.XML",
      "tif_img_path": "example/texture_image.tif",
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



Where:

* ``input_path``: Filepath to the input. Should either be a point cloud or a mesh.
* ``output_dir``: Directory path to the output folder where to save results.
* ``output_name`` (optional, default=```output_cars-mesh```): Name of the output mesh file (without extension)
* ``initial_state`` (optional, default= ```initial_pcd```): Initial state in the state machine. If you input a point cloud, it should be ```initial_pcd```. If you input a mesh, it could either be ```initial_pcd``` (you can compute new values over the points) or ```meshed_pcd``` (if for instance you only want to texture an already existing mesh).
* ``state_machine``: List of steps to process the input according to a predefined state machine (see below). Each step has three possible keys:``action`` (str) which corresponds to the trigger name, ``method`` (str) which specifies the method to use to do that step (possible methods are available in the ``/cars-mesh/param.py`` file, by default it is the first method that is selected), ``params`` (dict) which specifies in a dictionary the parameters for each method.

.. image:: ../images/fig_state_machine.png
    :alt: CARS-MESH State Machine

For each step, you can specify whether to save the intermediate output to disk.
To do so, in the step dictionary, you need to specify a key `save_output` as `true` (by default, it is `false`).
It will create a folder in the output directory named "intermediate_results" where these intermediate results will be saved.


If a texturing step is specified, then the following parameters become mandatory:

* ``rpc_path``: Path to the RPC xml file
* ``tif_img_path``: Path to the TIF image from which to extract the texture image
* ``utm_code``: The UTM code of the point cloud coordinates expressed as a EPSG code number for transformation purpose

Another parameter - optional - when applying a texture is the `image_offset`.
It is possible to use a cropped version of the image texture as long as the `image_offset` parameter is specified.
It is a tuple or a list of two elements (col, row) corresponding to the top left corner coordinates of the cropped image texture.
It will change the normalisation offset of the RPC data to make the texture fit to the point cloud.
If the image is only cropped on the bottom right side of the image, no offset information is needed.

Finally, you can run the following commands to activate the virtual environment and run the pipeline:

.. code-block:: bash

    source /venv/bin/activate
    cars-mesh /path/to/config_reconstruct.json
