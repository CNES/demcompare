.. _high_level_description:


Demcompare high level description
**********************************

Demcompare can be run through :ref:`demcompare_cli` that uses :ref:`demcompare_module`.

With an input configuration file, :ref:`demcompare_module` orchestrates :

* functions from :ref:`dem_tools_modules` for DEM manipulation

* functions from :ref:`coregistration_modules` for DEM coregistration

* functions from :ref:`stats_modules` to handle statistics metrics computation

* functions from :ref:`report_module` to create the output report (Work in progress)



.. figure:: /images/modules_schema.png
    :width: 800px
    :align: center

    Modules relationship.

Demcompare API is also detailed in `notebooks <https://github.com/CNES/demcompare/tree/master/notebooks>`_ and automatic API is generated in :doc:`/api_reference/index` section.


Demcompare conception
**********************

Demcompare architecture combines simple **python modules** with **python classes**. To generalize some parts, some of those classes have an **abstract architecture**.

Demcompare's abstraction are all implemented with the following structure:

1. The **class factory**, which is python file named like the class. It only handles the class object generation.
2. The **abstract class template**, which is a python file named like the class + "_template". This file includes all the abstract functions and attributes.
3. The **subclasses**, which are python files implementing the subclasses derived from the abstract class. With the class factory and the abstract class template, different subclasses can be implemented deriving from the abstract class template.

