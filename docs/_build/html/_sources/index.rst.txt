.. P2HNNS documentation master file, created by
   sphinx-quickstart on Thu Feb  8 02:22:38 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to P2HNNS's documentation!
==================================
P2HNNS is a Python library for efficient Point-to-hyperplane nearest neighbours search (P2HNNS) using locality sensitive hashing (LSH) approaches.
The library implements the 5 different methods described below.

- Bilinear Hyperplane (BH) hashing 
- Embedding Hyperplane (EH) hashing
- Multilinear Hyperplane (MH) hashing
- Nearest Hyperplane (NH) hashing
- Furthest Hyperplane (FH) hashing

The implementation is based on the original code of `HuangQiang <https://github.com/HuangQiang/P2HNNS>`_ (implemented in C++) and `stepping1st <https://github.com/stepping1st/hyperplane-hash/tree/master>`_ (implemented in Java).


Installation
============
The library can be installed via the pip package manager using the following command

.. code-block:: RST

   pip install P2HNNS

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
