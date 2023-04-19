.. _developer_install:

Developer Install
*****************

This package can be installed through the following commands:

.. code-block:: bash

    git clone https://github.com/CNES/demcompare
    cd demcompare
    make install
    source venv/bin/activate # to go in installed dev environment

Dependencies : **git**, **make**

The Makefile wrapper helps to install development environment and give the developer the commands to do a manual installation. 
By default, a python `virtualenv <https://docs.python.org/fr/3/library/venv.html>`_ is automatically created with the **make install** command. 

Please run **make help** command to see all the available commands.

It is also possible to change the virtualenv directory: 

.. code-block:: bash

    git clone https://github.com/CNES/demcompare
    cd demcompare
    VENV="other-venv-directory" make install
    source venv/bin/activate # to go in installed dev environment

.. note::
  Use **make clean** command to redo the install when environment state is undefined.


Sample execution from source
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Getting started example from source code of a basic DEM coregistration + statistics execution with the sample images and input configuration available on **demcompare** :

.. code-block:: bash

    cd data_samples # considering demcompare command is available (see above)
    demcompare sample_config.json # run demcompare example

- For more information about **demcompare**'s command line execution, please refer to: :ref:`command_line_execution`
- For more information about **demcompare**'s steps, please refer to: :ref:`coregistration`, :ref:`statistics`, :ref:`report`

