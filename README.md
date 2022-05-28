netpandas
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/chourmo/netpandas/workflows/CI/badge.svg)](https://github.com/chourmo/netpandas/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/chourmo/netpandas/branch/master/graph/badge.svg)](https://codecov.io/gh/chourmo/netpandas/branch/master)


Add graph functionnalities to a Pandas dataframe through its extension capabilities.

## Description

**Netpandas** adds graph functionalities on a standard pandas dataframe by a column of source-target node ids. Thus, the data model is edge-based, directed or not.
**Netpandas** is initialised with netpandas.set_network() function, with source and target column names and a directed boolean.
Graph functions are accessed behind a .net accessor.

### Example functions on a netpandas dataframe

. node properties : nodes, degree
. edge properties : duplicated, loops, 
. connected components : df.net.filter_by_components

**Netpandas** provides backends and conversion functions to Networkx, Scipy and Pandana.
**Netpandas** has functions to connect nodes based on spatial proximity. By using pandas accesor functionality, a Geopandas spatial dataframe can also be a graph dataframe __without__ subclassing Geopandas.

**Netpandas** has functions for arcs (list of edges) and path (list of nodes). The functions are useful to deal with shortest path graph functions.


### When not to use Netpandas

- when nodes have attributes
- when node ids are not integers
- when code is mostly base on the graph algorithms

## Import caveat ##

**Netpandans** uses the accessor functionality from pandas. Some operations in pandas may drop the net accessor. See ()
If one of these pandas function is used, the accessor must be added by netpandas.set_network()

### Documentation

Documentation is available at (http://http://netpandas.readthedocs.io/)

### Copyright

Copyright (c) 2022, chourmo


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
