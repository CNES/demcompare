.. _tuto_new_coregistration:

New coregistration class implementation
=======================================

Demcompare's architecture allows to easily implement a **new coregistration algorithm**.

To do so, a new class has to be implemented within the `demcompare/coregistration <https://github.com/CNES/demcompare/tree/master/demcompare/coregistration>`_ folder.
The new coregistration class inherits from the **CoregistrationTemplate** class  (see :ref:`coregistration_modules`) and must implement the following functions:

- The *__init__* and *fill_conf_and_schema* functions are necessary to initialize class attributes that are not already present in the **CoregistrationTemplate** class.
- The *_coregister_dems_algorithm* function is the function that perfoms the coregistration algorithm. It takes the two reprojected input dems as inputs, meaning that they will already have the
  same resolution and size. Both dems are **demcompare datasets** (see :ref:`dem_tools_modules`).
  The *_coregister_dems_algorithm* function must have the following outputs:

  1. The **Transformation** object containing the computed offsets.

  2. The **coregistered sec** and **coregistered ref** **demcompare datasets**. Those are the input datasets after the shifts of the coregistration algorithm have been applied. Please notice that those DEMs have been reprojected and coregistered, so they shall be used only for statistics computations such as its difference, error pdf, etc. For other applications performing *transformation.apply_transform(sec)* to the original secondary DEM is preferable.


.. note::
      Please notice that if crop and interpolations need to be done in the input DEM, then those should also be done on the input classification layers if present
      on the dem demcompare dataset in order to maintain the coherence between the dem and the classification.

- The *compute_results* function will do a logging of the obtained results and save them on the output **demcompare_results.json** file
- Other functions characteristic to the coregistration class may be implemented as well.

Hence, a basic *NewCoregistrationClass* would be implemented with the following structure :

.. code-block:: bash

    @Coregistration.register("new_coregistration_class")
    class NewCoregistrationClass(
        CoregistrationTemplate
    ):

    def __init__(self, cfg: ConfigType = None)
        """
        Return the coregistration object associated with the method_name
        given in the configuration

        Any coregistration class should have the following schema on
        its input cfg (optional parameters may be added for a
        particular coregistration class):

        coregistration = {
         "method_name": coregistration class name. str,
         "number_of_iterations": number of iterations. int,
         "sampling_source": optional. sampling source at which
           the dems are reprojected prior to coregistration. str
           "sec" (default) or "ref",
         "estimated_initial_shift_x": optional. estimated initial
           x shift. int or float. 0 by default,
         "estimated_initial_shift_y": optional. estimated initial
           y shift. int or float. 0 by default,
         "output_dir": optional output directory. str. If given,
           the coreg_dem is saved,
         "save_optional_outputs": optional. bool. Requires output_dir
           to be set. If activated, the outputs of the coregistration method
           (such as nuth et kaab iteration plots) are saveda and the internal
           dems of the coregistration
           such as reproj_dem, reproj_ref, reproj_coreg_sec,
           reproj_coreg_ref are saved.
        }

        :param cfg: configuration {'method_name': value}
        :type cfg: ConfigType
        """

    def fill_conf_and_schema(self, cfg: ConfigType = None) -> ConfigType:
        """
        Add default values to the dictionary if there are missing
        elements and define the configuration schema

        :param cfg: coregistration configuration
        :type cfg: ConfigType
        :return cfg: coregistration configuration updated
        :rtype: ConfigType
        """


    def _coregister_dems_algorithm(
        self,
        sec: xr.Dataset,
        ref: xr.Dataset,
    ) -> Tuple[Transformation, xr.Dataset, xr.Dataset]:
        """
        Coregister_dems, computes coregistration
        transform and coregistered DEMS of two DEMs
        that have the same size and resolution.

        :param sec: sec xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layers : 3D (row, col, indicator) xr.DataArray
        :type sec: xarray Dataset
        :param ref: ref xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layers : 3D (row, col, indicator) xr.DataArray
        :type ref: xarray Dataset
        :return: transformation, reproj_coreg_sec xr.DataSet,
                 reproj_coreg_ref xr.DataSet. The xr.Datasets containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layers : 3D (row, col, indicator) xr.DataArray
        :rtype: Tuple[Transformation, xr.Dataset, xr.Dataset]
        """

    def compute_results(self):
        """
        Save the coregistration results on a Dict
        The altimetric and coregistration results are saved.
        Logging of the altimetric results is done in this function.

        :return: None
        """




The **Transformation** is the object storing the coregistration offsets, and can be created the following way:

.. code-block:: bash

    transform = Transformation(
                x_offset=x_offset,
                y_offset=y_offset,
                z_offset=z_offset,
                estimated_initial_shift_x=self.estimated_initial_shift_x,
                estimated_initial_shift_y=self.estimated_initial_shift_y,
                adapting_factor=self.adapting_factor,
            )

The *adapting_factor* shows if the coregistration has been performed at a resolution different from the
original **sec** resolution (if the *sampling_source* parameter was set to *ref* (see :ref:`coregistration`).

