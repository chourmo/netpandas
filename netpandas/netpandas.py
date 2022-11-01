import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_list_like

from . import backends, spatial, graph


EXT_NAME = "net"
NODE_ID_NAME = "nodeid"
EDGE_NAME = "edge"


# --------------------
# Net dataframe creation functions
# --------------------


def set_network(
    df, directed=False, source=None, target=None, edge=None, edge_name=EDGE_NAME, drop=True
):
    """
    Set network attributes in a dataframe

    Args:
        directed : edges are in both directions, default to False,
                   arcs are from source to target
        source : source column name, must contain integers
        target : target column name, must contain integers
        edge : column of tuples, alternative creation parameters to source and target
    """

    if (source is None or target is None) and edge is None:
        raise ValueError("source and target, or edge must be set to a column name")

    if type(directed) != bool:
        raise ValueError("directed must be a boolean")

    ndf = df.copy(deep=True)
    ndf.attrs[EXT_NAME] = {}
    ndf.attrs[EXT_NAME]["directed"] = directed

    if source in df.columns and target in df.columns:
        ndf[edge_name] = _nodes_to_edge(ndf[source], ndf[target])
        ndf.attrs[EXT_NAME]["name"] = edge_name

        if drop:
            ndf = ndf.drop(columns=[source, target])
    elif edge is not None and edge in df.columns and is_list_like(df[edge]):
        ndf.attrs[EXT_NAME]["name"] = edge

    elif edge is not None and edge not in df.columns or not is_list_like(df[edge]):
        raise ValueError("{0} must be a column of tuples".format(edge))
    else:
        raise ValueError(
            "{0} and {1} must be columns of integers".format(source, target)
        )

    return ndf


def is_netdf(df):
    """
    check if dataframe as attributes and columns for using netpandas functions
    """

    if EXT_NAME not in df.attrs:
        return False

    if not all(k in df.attrs[EXT_NAME] for k in ["name", "directed"]):
        return False

    if df.attrs[EXT_NAME]["name"] not in df.columns:
        return False

    if not is_list_like(df.net.edges):
        return False

    return True


@pd.api.extensions.register_dataframe_accessor(EXT_NAME)
class NetworkAccessor(
    backends.Backends, spatial.SpatialFunctions, graph.GraphFunctions
):
    """
    Dataframe extension adding network support to a dataframe
    """

    # ----------------------------
    # initialisation
    # ----------------------------

    def __init__(self, pandas_obj):

        self._obj = pandas_obj

    @property
    def name(self):
        """name of the network columns"""
        # return [self._obj.attrs[EXT_NAME]["name"]]
        return self._obj.attrs[EXT_NAME]["name"]

    @property
    def directed(self):
        """store direction property in dataframe metadata"""
        return self._obj.attrs[EXT_NAME]["directed"]

    # ----------------------------
    # network attributes
    # ----------------------------

    @property
    def nodes(self):
        """
        return Series of nodes without attributes
        """
        a = self.named_numpy_edges("_s", "_t")
        return pd.Series(np.unique(np.vstack((a["_s"], a["_t"]))))

    @property
    def edges(self):
        """
        returns a Series of edges without attributes
        """
        return self._obj[self.name]

    def named_numpy_edges(self, source="source", target="target"):
        """
        returns a Series of edges without attributes
        """
        n = self.edges.to_numpy()
        n = np.array(n, dtype=[(source, "i8"), (target, "i8")])
        return n

    @property
    def sources(self):
        """
        returns a Series of sources node ids with same index as dataframe
        """
        return pd.Series(
            data=self.named_numpy_edges("source", "target")["source"],
            index=self._obj.index,
        )

    @property
    def targets(self):
        """
        returns a Series of targets node ids with same index as dataframe
        """
        return pd.Series(
            data=self.named_numpy_edges("source", "target")["target"],
            index=self._obj.index,
        )

    def reversed_edges(self):
        """
        returns a Series of edges without attributes and
        with reversed sources and targets
        """
        return self.edges.apply(lambda x: tuple(x)[::-1])

    def expand_edges(self, source="source", target="target", drop=False):
        """
        returns a netpandas dataframe with new columns of
        sources and targets, names source and target"""
        df = self._obj.copy()
        if source in df and target in df:
            return df
        df[source] = self.sources
        df[target] = self.targets
        
        if drop:
            df = df.drop(columns=[self.name])
        
        return df

    # ----------------------------
    # network properties
    # ----------------------------

    def has_duplicated_edges(self):
        """
        returns True if some edges are duplicated
        """
        return len(self.duplicated_edges()) > 0

    def duplicated_edges(self):
        """
        returns dataframe of duplicated edges
        """
        df = self.edges
        return df.loc[df.duplicated()]

    def has_loops(self):
        """
        returns True if some edges source and target is the same
        """
        return len(self.loops) > 0

    def loops(self):
        """
        returns dataframe of self loop edges
        """
        return self.edges.loc[self.is_loop]

    def is_loop(self):
        """
        returns dataframe of self loop edges
        """
        return self.sources == self.targets

    def self_loops(self, refid, num=None, min_num=None):
        """
        returns a Series of self crossing nodes, sharing a refid value
        """
        df = self.node_column_degree(refid)

        if num is not None:
            df = df.loc[df == num]
        elif min_num is not None:
            df = df.loc[df > min_num]
        else:
            raise ValueError("num or min_num must have a value")
        return df.to_frame("_size").reset_index(NODE_ID_NAME)[NODE_ID_NAME]

    def is_complex_selfloop(self, refid):
        """
        Returns a boolean Series of self-loops, excluding simple loops as roundabouts
        """
        nodes = self.self_loops(refid, min_num=2)
        return self._obj[refid].isin(nodes.index)

    def is_simple_selfloop(self, refid):
        """
        Returns a boolean Series of simple self-loops as roundabouts
        """
        df = self.expand_edges(source="_s", target="_t")
        nodes = self.self_loops(df, refid, num=2)

        # loop if all nodes in refid are of degree 2
        nodes = nodes.groupby(refid).size().to_frame("_degree")
        nodes["_ref"] = df.groupby(refid)["_s"].size()
        nodes = nodes.loc[nodes._degree == nodes._ref]

        return self._obj[refid].isin(nodes.index)

    def degree(self):
        """
        returns a Series with node id as index and number of arcs connected
        """
        a = self.named_numpy_edges("_s", "_t")
        arr, c = np.unique(np.vstack((a["_s"], a["_t"])), return_counts=True)
        return pd.Series(data=c, index=arr)

        # df = pd.concat([self.sources, self.targets], ignore_index=True)
        # return df.value_counts().sort_index()

    def degree_in(self):
        """
        returns a Series with node id as index and
        number of arcs connected to node
        """
        return self.targets.value_counts()

    def degree_out(self):
        """
        returns a Series with node id as index and
        number of arcs connected from node
        """
        return self.sources.value_counts()

    def max_node_value(self):
        """
        returns the max integer value of node ids
        """
        return max(self.sources.max(), self.targets.max())

    def node_column_degree(self, col, source=None, target=None):
        """
        Get the node degree for a subgraph sharing values in a column
        return a Series with index column value / node id, values node degree
        """
        if source is None or target is None:
            source = "_s"
            target = "_t"
            df = self.expand_edges(source, target)
        else:
            df = self._obj.copy()

        df = df[[col, source, target]]
        df = pd.concat(
            [
                df[[col, "_t"]].rename(columns={"_t": NODE_ID_NAME}),
                df[[col, "_s"]].rename(columns={"_s": NODE_ID_NAME}),
            ]
        )
        return df.groupby([col, NODE_ID_NAME]).size()

    def node_column_degree_inout(self, col, source=None, target=None):
        """
        Get the node degree for a subgraph sharing values in a column
        return a Dataframe with index column value / node id,
        values node degree for source and target
        """

        if source is None or target is None:
            source = "_s"
            target = "_t"
            df = self.expand_edges(source, target)
        else:
            df = self._obj.copy()

        df = pd.concat([df[[col, source]], df[[col, target]]])
        df[NODE_ID_NAME] = df[source]
        df[NODE_ID_NAME] = df[NODE_ID_NAME].fillna(df[target]).astype(int)
        df = df.groupby([col, NODE_ID_NAME]).agg(
            source=(source, "count"), target=(target, "count")
        )
        return df

    def node_sharedcolumn(self, col):
        """
        returns the number of unique values in col shared by nodes, node id in index
        """

        df = self.expand_edges(source="_s", target="_t")[[col, "_s", "_t"]]
        df = pd.concat(
            [
                df[["_s", col]],
                df[["_t", col]].rename(columns={"_t": "_s"}),
            ],
            ignore_index=True,
        )

        return df.groupby("_s").nunique(col)[col]

    def has_any_nodes(self, nodes):
        """
        returns a boolean series if nodes are in an edge source or target
        """

        return self.sources.isin(nodes) | self.targets.isin(nodes)

    def has_all_nodes(self, nodes):
        """
        returns a boolean series if node is in an edge source and target
        """

        return self.sources.isin(nodes) & self.targets.isin(nodes)

    def align_nodes(self, nodes, method="edges"):
        """
        align edges on self index
        """

        # if method is None return pd.Series with nodes as values
        if method is None:
            return nodes

        df = self.expand_edges(source="source", target="target")

        # convert to edges form
        df = pd.merge(
            df,
            nodes.to_frame("_s"),
            left_on="source",
            right_index=True,
            how="left",
        )
        df = pd.merge(
            df,
            nodes.to_frame("_t"),
            left_on="target",
            right_index=True,
            how="left",
        )

        if method == "edges":
            return df[["_s", "_t"]]

        # else as if method is aggregate method

        return None

    def renumber_nodes(self, nodes=None):
        """
        change source and target values in nodes array to new values
        if nodes is None, renumber all nodes, starting at 0
        returns a new edge column
        """

        if nodes is None:
            n = self.nodes
            m = pd.Series(data=range(0, len(n)), index=n.values)
        else:
            m = self.max_node_value()
            m = pd.Series(data=range(m + 1, m + 1 + len(nodes)), index=nodes)

        # expand edges to sources and targets
        df = self.expand_edges(source="_s", target="_t", drop=True)

        df["_s"] = df["_s"].map(m).fillna(df["_s"]).astype(int)
        df["_t"] = df["_t"].map(m).fillna(df["_t"]).astype(int)

        df = set_network(
            df, directed=self.directed, source="_s", target="_t", edge_name="new_edge"
        )

        return df["new_edge"]

    def adjency_list(self, direction, index=False):
        """
        returns adjency lists, indexed on nodes :

        if direction ='in', returns edges into node
        if direction = 'out', returns edges from node
        if direction = 'both', returns a dataframe with 2 columns, in and out

        if index is True, returns a tuple of tuples of (index, node), else returns a tuple of nodes
        """

        df = self.expand_edges(source="_s", target="_t")[["_s", "_t"]]

        if index:
            if df.index.name is None:
                df.index.name = "_ix"
            ix = df.index.name
            df = df.reset_index()

        if direction == "in" or direction == "both":
            if index:
                df["_next"] = list(zip(df[ix], df["_t"]))
                res = df.groupby("_s")["_next"].agg(tuple)

            else:
                res = df.groupby("_s")["_t"].agg(tuple)

            res.index.name = None
            if direction == "in":
                return res
            else:
                res = res.to_frame("in")

        # direction is out or both
        if index:
            df["_next"] = list(zip(df[ix], df["_s"]))
            res2 = df.groupby("_t")["_next"].agg(tuple)

        else:
            res2 = df.groupby("_t")["_s"].agg(tuple)

        res2.index.name = None
        if direction == "out":
            return res2
        else:
            return pd.merge(
                res,
                res2.to_frame("out"),
                how="outer",
                left_index=True,
                right_index=True,
            )

        return res

    def to_directed(
        self,
        direction=None,
        backward_value=-1,
        bidi_value=2,
        forward=None,
        backward=None,
        split_mid_nodes=False,
    ):
        """
        Convert an unidrected netdataframe to a directed one
        if ndf is spatially enabled, the geometry is reversed

        Args:
            direction, a column name containing a bidi_value or backward value,
              all other values ar for strictly forward edges
            forward and backward, column names dropping na values edges
            split_mid_nodes : boolean, change node value if nodes are connected to 2 arcs

        Returns :
            a directed netpandas dataframe with same content and index as ndf,
            except direction, forward and backward

        """

        if self.directed:
            raise ValueError("ndf must not be directed")

        if self.is_spatial:
            geo_col = self._obj.geometry.name

        df = self._obj.copy()

        if split_mid_nodes:
            nodes = self.degree()
            nodes = nodes.loc[nodes == 2]

        if direction is not None:
            inv_df = df.loc[
                (df[direction] == bidi_value) | (df[direction] == backward_value)
            ].copy()
            df = df.loc[df[direction] != backward_value].copy()

        elif forward is not None and backward is not None:
            inv_df = df.loc[~df[backward].isna()].copy()
            inv_df = inv_df.drop(columns=[forward])
            inv_df = inv_df.rename(columns={backward: forward})

            df = df.loc[~df[forward].isna()].copy()
            df = df.drop(columns=[backward])

        else:
            raise ValueError("dir_col or forward_col and backward_col must be set")

        # reverse source and target
        inv_df[self.name] = inv_df.net.reversed_edges()

        if self.is_spatial:
            inv_df[geo_col] = spatial.reverse(inv_df[geo_col])

        df = pd.concat([df, inv_df])
        df = set_network(df, directed=True, edge=self.name)

        if split_mid_nodes:
            df = _split_mid_nodes(df, nodes)

        if self.is_spatial:
            df = df.set_geometry(geo_col, crs=self._obj.crs)

        return df.reset_index(drop=True)


# -----------------------------------
# nodes and edges internal functions


def _nodes_to_edge(df1, df2):
    """Convert a dataframe to a list like"""

    if not is_integer_dtype(df1):
        raise ValueError("sources must be integers")
    if not is_integer_dtype(df2):
        raise ValueError("targets must be integers")

    return pd.Series(zip(df1.to_numpy(), df2.to_numpy()), index=df1.index)


def _split_mid_nodes(ndf, nodes):
    """
    renumber nodes of reversed edges, reversed edges must share index values
    """

    edge = ndf.net.name
    res = ndf.net.expand_edges("_s", "_t", drop=True)
    res.index.name = "_uid"
    res = res.reset_index()
    start = ndf.net.max_node_value() + 1
    n = nodes.index

    # remap one source, mapper index is edge id, value old node id
    # remap only duplicated indexes, non duplicated are oneways
    mapper = res.loc[(res._s.isin(n)) & (res._uid.duplicated()), ["_s", "_uid"]]
    mapper = mapper.drop_duplicates(subset=["_s"])
    mapper["_new_s"] = range(start, start + len(mapper))

    res = pd.merge(res, mapper, on=["_s", "_uid"], how="left")

    # remap target in nodes and different edge id
    mapper = mapper.rename(columns={"_s": "_t", "_new_s": "_new_t", "_uid": "_uid_t"})
    res = pd.merge(res, mapper, on="_t", how="left")

    # remap node ids
    res.loc[(~res._new_s.isna()), "_s"] = res["_new_s"]
    res.loc[(~res._uid_t.isna()) & (res._uid != res._uid_t), "_t"] = res["_new_t"]

    res["_s"] = res["_s"].astype(int)
    res["_t"] = res["_t"].astype(int)

    res = res.drop(columns=["_uid", "_uid_t", "_new_s", "_new_t"])
    res.index = ndf.index

    return set_network(res, source="_s", target="_t", directed=True)