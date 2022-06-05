"""Add graph functionnalities to a Pandas dataframe through its extension capabilities."""

# Add imports here
from .netpandas import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions


# main API functions

from netpandas.netpandas import set_network, is_netdf
from netpandas.spatial import connect_geodataframe
from netpandas.path import *
from netpandas.graph import connect_nodes, topological_edge_index
from netpandas.arc import merge_arcs, sort_arcs, no_self_loop_index

# backends

from netpandas.networkx_backend import *
from netpandas.scipy_backend import *
from netpandas.pandana_backend import *
from . import _version
__version__ = _version.get_versions()['version']
