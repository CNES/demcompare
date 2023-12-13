.. _tuto_new_dem_processing:

New DEM processing method implementation
========================================

Demcompare's architecture allows to easily implement a **new DEM processing method computation**.

To do so, a new class has to be implemented within `demcompare/dem_processing/dem_processing_methods.py <https://github.com/CNES/demcompare/blob/master/demcompare/dem_processing/dem_processing_methods.py>`_ file, according to
the new DEM processing method's structure (see :ref:`stats_modules`).


Basic DEM processing method structure and functions
***************************************************

The new DEM processing method class inherits from the **DemProcessingTemplate** class and must implement the **process_dem** function. This
function takes two *xr.Dataset* as an entry and performs the corresponding DEM processing computation on the datasets. The output should be a *xr.Dataset*.

One may also implement the **__init__** function of the new DEM Processing class, mostly if this DEM Processing contains class attributes.

Hence, a basic *NewDemProcessingClass* would be implemented with the following structure :

.. code-block:: bash

    @DemProcessing.register("new_dem_processing_class")
    class NewDemProcessingClass(DemProcessingTemplate):

        # Optional, only needed if the DEM processing object has its own parameters
        def __init__(self, parameters: Dict = None):
            """
            Initialization the DEM Processing object

            :param parameters: optional input parameters
            :type parameters: dict
            :return: None
            """

        def process_dem(
            self,
            dem_1: xr.Dataset,
            dem_2: xr.Dataset,
        ) -> xr.Dataset: