.. idl documentation master file, created by
   sphinx-quickstart on Thu Jan 23 23:56:54 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to IDL Documentation
===========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/models
   api/sim
   api/attention
   examples

IDL (Implicit Deep Learning) is a Python package that implements implicit deep learning models (with specialized attention mechanisms) and state-driven implicit models.

Key Features
-----------

* Implicit Models with optional low-rank approximation
* State-driven Implicit Models (SIM)
* Implicit Attention Mechanisms
* Support for various datasets and architectures

Quick Install
------------

.. code-block:: bash

   pip install idl

Install from source:

.. code-block:: bash

   git clone https://github.com/HoangP8/Implicit-Deep-Learning
   cd Implicit-Deep-Learning
   pip install -e .