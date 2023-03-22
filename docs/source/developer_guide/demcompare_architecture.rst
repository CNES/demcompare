.. _demcompare_architecture:
  
Demcompare architecture
=======================

This section gives insights of demcompare architecture and design to help understand how the software works, in order to ease 
developer contributions. 

The following figure show demcompare architecture organization. 

SCHEMA TODO explaining below blocks


Demcompare high level description
**********************************

Demcompare can be run through :ref:`demcompare_cli` that uses :ref:`demcompare_module`. 

With input configuration file, :ref:`demcompare_module` orchestrates functions from :ref:`dem_tools_modules` for dem manipulation,
from :ref:`coregistration_modules` for dem coregistration and :ref:`stats_modules` to handle statistics metrics computation.

A report is generated from :ref:`report_module` (Work in progress).

Demcompare API is also demonstrated in more details in notebooks and autoAPI is generated in API reference section.

The following sections at the bottom give details for each subpart. 

Demcompare conception
**********************

Demcompare's architecture combines simple python modules with python classes. To add genericity to some parts, some of those classes have an abstract architecture.

Demcompare's abstract classes are all implemented with two main python files:

- The class factory, which is python file named like the class. It only handles the class object generation.
- The abstract class template, which is a python file named like the class + "_template". This file includes all the abstract functions and attributes.

With the class factory and the abstract class template, the different subclasses can be implemented:

- The subclasses, which are python files implementing the subclasses derived from the abstract class. 


TODO: check if elements are missing for global demcompare design understanding

.. note::

    Please contribute if elements are missing or are unclear to enter demcompare code.

.. toctree::
  :maxdepth: 2

  architecture/demcompare_cli.rst
  architecture/demcompare_module.rst
  architecture/dem_tools_modules.rst
  architecture/coregistration_modules.rst
  architecture/stats_modules.rst
  architecture/report_module.rst



