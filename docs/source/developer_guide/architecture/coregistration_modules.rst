.. _coregistration_modules:


Coregistration modules
======================

This section explains coregistration modules in demcompare. 

In the following image we can find the classes that take part in demcompare's coregistration step, along
with their relationship.

Coregistration step architecture
--------------------------------

The  `coregistration_class`_ and `transformation_class`_ handle the API for the dem coregistration. The :ref:`demcompare_module`
creates a `coregistration_class`_ object. The `coregistration_class`_ creates a `transformation_class`_ object when the coregistration offsets are
obtained.

The `transformation_class`_ object is in charge of storing the offsets and applying them to the secondary dem.

.. figure:: /images/schema_coregistration_class.png
    :width: 400px
    :align: center

    Coregistration classes relationship.

Coregistration
**************

.. _coregistration_class:

The coregistration class in demcompare has the following structure:

- **Coregistration**: The class Factory in `Coregistration file <https://github.com/CNES/demcompare/blob/master/demcompare/coregistration/coregistration.py>`_

- **CoregistrationTemplate**: The abstract class in `CoregistrationTemplate file <https://github.com/CNES/demcompare/blob/master/demcompare/coregistration/coregistration_template.py>`_

- **NuthKaabInternal**: Nuth et kaab coregistration algorithm in `NuthKaabInternal file <https://github.com/CNES/demcompare/blob/master/demcompare/coregistration/nuth_kaab_internal.py>`_

A Coregistration object is in charge of computing the offsets between two DEMs that have the same resolution and size, giving as an output
a **Transformation** object, along with the two reprojected and coregistered dems.

It is to be noticed that to compute the offsets between two DEMs, they need to have the same resolution and size. For this reason, the **coregistration**
module perfoms a reprojection using the :ref:`dem_tools_modules` API.

One can find here the full list of API functions available in the `coregistration_class`_, as well as their description and
input and output parameters:
`Coregistration API <https://demcompare.readthedocs.io/en/latest/api_reference/demcompare/coregistration/coregistration_template/index.html>`_

For information on how to create a new coregistration class, please see :ref:`tuto_new_coregistration`.


Transformation
**************

.. _transformation_class:

-  **Transformation** class in `Transformation file <https://github.com/CNES/demcompare/blob/master/demcompare/transformation.py>`_

The Transformation class stores the offsets obtained during the coregistration step. It also has the API to apply the
offsets to an input DEM. It is created by the Coregistration class and given as an output.

One can find here the full list of API functions available in the `transformation_class`_, as well as their description and
input and output parameters:
`Transformation API <https://demcompare.readthedocs.io/en/latest/api_reference/demcompare/transformation/index.html>`_