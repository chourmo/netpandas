# main API functions

from netpandas.netpandas import set_network as set_network
from netpandas.netpandas import is_netdf as is_netdf

from netpandas.spatial import connect_geodataframe as connect_geodataframe

from netpandas.path import expand_paths as expand_paths
from netpandas.path import paths_distance as paths_distance
from netpandas.path import make_path as make_path
from netpandas.path import add_paths as add_paths

from netpandas.graph import connect_nodes as connect_nodes
from netpandas.graph import topological_edge_index as topological_edge_index

from netpandas.arc import merge_arcs as merge_arcs
from netpandas.arc import sort_arcs as sort_arcs
from netpandas.arc import no_self_loop_index as no_self_loop_index

# backends

from netpandas.networkx_backend import Networkx_backend as Networkx_backend
from netpandas.scipy_backend import  Scipy_backend as  Scipy_backend
from netpandas.pandana_backend import Pandana_backend as Pandana_backend
