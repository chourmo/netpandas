# -*- coding: utf-8 -*-
# graph functions

import numpy as np
import scipy.sparse as ss
import pandas as pd
import netpandas as npd
from pandas.api.types import is_integer_dtype


class GraphFunctions:

    # Component functions

    def node_components(self, strong=True):
        """
        Find a component number for each node
        joint nodes share the same component number
        """

        # scipy prefers continuous integer node values
        # use intermediary node values

        ndf = self._obj.copy()
        ndf["_new_edge"] = self.renumber_nodes()
        ndf = npd.set_network(ndf, directed=self.directed, edge="_new_edge")

        G = ndf.net.graph("scipy")

        if strong:
            connection = "strong"
        else:
            connection = "weak"

        comp, nodes = ss.csgraph.connected_components(
            csgraph=G, directed=self.directed, connection=connection, return_labels=True
        )

        # convert back to original node values
        return pd.Series(data=nodes, index=self.nodes.values)

    def edge_components(self, strong=True):
        """
        Find source and target component number for each edge
        joint edges share the same component number
        """

        nodes = self.node_components(strong=strong)
        res = self.align_nodes(nodes, method="edges")

        return res

    def filter_by_component(self, size=None, strong=True):
        """
        Filter graph by a minimum component size or bigest component

        Arguments :
            size : minimum number of nodes in a component, if None keep largest component
            strong : if directed graph and True, use strongly connected components

        Results :
            a filtered netpandas graph with filtered edges
        """

        nodes = self.node_components(strong=strong)
        cc = nodes.value_counts().sort_values(ascending=False)

        if size is None:
            cc = cc.head(1)
        else:
            cc = cc.loc[cc >= size]

        nodes = nodes.loc[nodes.isin(cc.index)].sort_index()
        return self._obj.loc[self.sources.isin(nodes.index)].copy()


def connect_nodes(ndf, nodes, fill_values=None, ending="start"):
    """
    add edges to connect list of nodes to a new node
    if ending='start', add edges from a new node to the list of nodes
    if ending='end', add edges from the list of nodes to a new node
    fill columns by fill_values if not None, see Dataframe.fillna value

    returns a tuple of a new netpandas dataframe and new node number
    """

    if isinstance(nodes, list):
        if len(nodes) == 0:
            return ndf, pd.NA
        else:
            node_list = nodes.copy()
    elif isinstance(nodes, pd.Series) and is_integer_dtype(nodes):
        node_list = nodes.drop_duplicates().to_list()
    else:
        raise TypeError("nodes must be lists or Series of integers")

    graph = ndf.copy()
    n = ndf.net.max_node_value() + 1

    if ending == "start":
        edges = pd.DataFrame({"_s": [n] * len(node_list), "_t": node_list})

    elif ending == "end":
        edges = pd.DataFrame({"_s": node_list, "_t": [n] * len(node_list)})

    edges = npd.set_network(edges, source="_s", target="_t", directed=ndf.net.directed)
    graph = pd.concat([graph, edges], ignore_index=True)

    if fill_values is not None:
        graph = graph.fillna(fill_values)

    return graph, n


def disconnect_nodes(ndf, nodes, drop=True):
    """
    return a new netdataframe with unique nodes values,
    so that edges are not connected
    if drop, drop source and target columns
    """

    res = ndf.net.expand_edges(source="_s", target="_t", drop=True)

    # disconnect sources
    m = ndf.net.max_node_value + 1
    data = np.arange(m, m + res.shape[0])
    res.loc[res._s.isin(nodes), "_s"] = pd.Series(data=data, index=res.index)

    # disconnect sources
    m = data.max() + 1
    data = np.arange(m, m + res.shape[0])
    res.loc[res._t.isin(nodes), "_t"] = pd.Series(data=data, index=res.index)

    res = npd.set_network(
        res, directed=ndf.net.directed, source="_s", target="_t", drop=drop
    )

    return res


def topological_edge_index(ndf, refid, source=None, target=None):
    """
    Returns a Series of increasing integers,
    values changing if refid changes or on crossing nodes (degree not 2)
    """
    if source is None or target is None:
        source = "_s"
        target = "_t"
        res = ndf.net.expand_edges(source, target)

    else:
        res = ndf.copy()

    nodes = ndf.net.degree()
    nodes = nodes.loc[nodes != 2]

    df_n = (res._s.isin(nodes.index)) | (res[refid] != res[refid].shift(1))

    return df_n.cumsum() - 1
