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

