.. _coregistration_modules:


Coregistration modules
======================

This section explains coregistration modules in demcompare. 

In the following image we can find the classes that take part in demcompare's coregistration step, along
with their relationship.

Coregistration step architecture
--------------------------------

The **coregistration.py** and **transformation.py** modules handle the API for the dem coregistration. The :ref:`demcompare_module`
creates a **Coregistration** object. The **Coregistration** object creates a **Transformation** when the coregistration offsets are
obtained.

The **Transformation** object is in charge of storing the offsets and applying them to the secondary dem.

.. figure:: /images/schema_coregistration_class.png
    :width: 400px
    :align: center

    Coregistration classes relationship.

Coregistration
**************

The coregistration class in demcompare has the following structure:

- **Coregistration**: The class Factory. Implemented in `coregistration/coregistration.py`
- **CoregistrationTemplate**: The abstract class. Implemented in `coregistration/coregistration_template.py`
- **NuthKaabInternal**: Nuth et kaab coregistration algorithm. Implemented in `coregistration/nuth_kaab_internal.py`

The coregistration class computes the offsets between two DEMs that have the same resolution and size, giving as an output
a Transformation object, along with the two reprojected and coregistered dems.

Transformation
**************

- **Transformation**: Implemented in `transformation.py`

The Transformation class stores the offsets obtained during the coregistration step. It also has the API to apply the
offsets to an input DEM. It is created by the Coregistration class and given as an output.
