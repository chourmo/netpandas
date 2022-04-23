# -*- coding: utf-8 -*-

import pandas as pd

from . import backends

try:
    import networkx as nx

    NX_GRAPH = type(nx.Graph())
    NX_DIGRAPH = type(nx.DiGraph())
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


NETWORKX_NAME = "networkx"


class Networkx_backend(backends.Backends):
    """
    Networkx backend
    """

    @backends.Backends.graph.register(NETWORKX_NAME)
    def nx_graph(self, backend, attributes=None):
        """
        create a networkx xgraph
        """

        if self.directed:
            nxtype = nx.DiGraph
        else:
            nxtype = nx.Graph

        df = self.expand_edges(source="_s", target="_t")

        G = nx.from_pandas_edgelist(
            df,
            source="_s",
            target="_t",
            edge_attr=attributes,
            create_using=nxtype,
        )

        return G

    @backends.Backends.extract_edges.register(NX_GRAPH)
    @backends.Backends.extract_edges.register(NX_DIGRAPH)
    def nx_edges(self, graph, source="source", target="target", attributes=None):
        """
        extract networkx edges
        """
        df = nx.to_pandas_edgelist(graph, source=source, target=target)

        if attributes is not None:
            df = df[[source, target] + attributes]

        return self._align_edges(df)

    @backends.Backends.extract_nodes.register(NX_GRAPH)
    @backends.Backends.extract_nodes.register(NX_DIGRAPH)
    def nx_nodes(self, graph, attributes=None):
        """
        extract nodes dataframe in graph
        """

        if attributes is None:
            return pd.Series(data=graph.nodes())
        else:
            attrs = {}
            for c in attributes:
                attrs[c] = pd.Series(nx.get_node_attributes(graph, c))

            return pd.concat(attrs, axis=1)
