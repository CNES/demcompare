.. _user_guide:

============
Command line
============

You can run too functions with the ``mesh_3d`` cli:

* ``mesh_3d reconstruct`` launches the 3D reconstruction pipeline according to the user specifications
* ``mesh_3d evaluate`` computes metrics between two point clouds and saves visuals for qualitative analysis

Reconstruct
===========

To launch the 3D reconstruction pipeline, you first need to configure it in a JSON file like the following:

.. code-block:: JSON

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


Where:

* ``input_path``: Filepath to the input. Should either be a point cloud or a mesh.
* ``output_dir``: Directory path to the output folder where to save results.
* ``initial_state`` (optional, default= ```initial_pcd```): Initial state in the state machine. If you input a point cloud, it should be ```initial_pcd```. If you input a mesh, it could either be ```initial_pcd``` (you can compute new values over the points) or ```meshed_pcd``` (if for instance you only want to texture an already existing mesh).
* ``state_machine``: List of steps to process the input according to a predefined state machine (see below). Each step has three possible keys:``action`` (str) which corresponds to the trigger name, ``method`` (str) which specifies the method to use to do that step (possible methods are available in the ``/mesh_3d/param.py`` file, by default it is the first method that is selected), ``params`` (dict) which specifies in a dictionary the parameters for each method.

.. image:: images/fig_state_machine.png
    :alt: Mesh 3D State Machine


If a texturing step is specified, then the following parameters become mandatory:

* ``rpc_path``: Path to the RPC xml file
* ``tif_img_path``: Path to the TIF image from which to extract the texture image
* ``utm_code``: The UTM code of the point cloud coordinates expressed as a EPSG code number for transformation purpose


Finally, you can launch the following commands to activate the virtual environment and run the pipeline:

.. code-block:: bash

    source /venv/bin/activate
    mesh_3d reconstruct /path/to/config.json


Evaluate
========

The evaluation function computes a range of metrics between two point clouds and outputs visuals for
qualitative analysis.

Configure the pipeline in a JSON file like the following:

.. code-block:: JSON

    {
      "input_path_1": "/path/to/point_cloud/or/mesh_1.ply",
      "input_path_2": "/path/to/point_cloud/or/mesh_2.ply",
      "output_dir": "/path/to/output_dir"
    }


Where:

* ``input_path_1``: Filepath to the first input. Should either be a point cloud or a mesh.
* ``input_path_2``: Filepath to the second input. Should either be a point cloud or a mesh.
* ``output_dir``: Directory path to the output folder where to save results.

Finally, you can launch the following commands to activate the virtual environment and run the evaluation:

.. code-block:: bash

    source /venv/bin/activate
    mesh_3d evaluate /path/to/config.json
