# -*- coding: utf-8 -*-

import pandas as pd
import netpandas as npd

import netpandas.spatial as sp


NODE_ID_NAME = "nodeid"


# -------------------
# Arc  functions, arcs are edges sharing a common id column
# -------------------


def merge_arcs(ndf, arcid, source=None, target=None):
    """
    Merge edges that share an arcid,
    all column must have the same values for an arcid,
    arc edges must be ordered, first source and last target are source and target of an arc

    Args:
        ndf : a non directd streetpy dataframe
        arcid : an id or arcs to merge

    Returns:
        a new streetpy dataframe with a new index
    """
    if source is None or target is None:
        source = "_s"
        target = "_t"
        res = ndf.net.expand_edges(source, target, drop=True)
    else:
        res = ndf.copy()

    res_s = res[[arcid, source]].drop_duplicates(subset=arcid, keep="first")
    res_t = res[[arcid, target]].drop_duplicates(subset=arcid, keep="last")
    res_geoms = sp.linemerge(res.set_index(arcid)[ndf.geometry.name])

    res = res.drop(columns=[source, target, ndf.geometry.name])
    res = res.drop_duplicates(subset=arcid)

    # merge source, target, geometry
    res = pd.merge(res, res_s, on=arcid, how="left")
    res = pd.merge(res, res_t, on=arcid, how="left")
    res = pd.merge(
        res,
        res_geoms.to_frame(ndf.geometry.name),
        left_on=arcid,
        right_index=True,
        how="left",
    )

    res = res.set_geometry(ndf.geometry.name, crs=ndf.crs)

    # MultineString arcs should not have been merged
    res = res.reset_index(drop=True).explode(index_parts=True).reset_index(drop=True)

    res = npd.set_network(
        res,
        directed=ndf.net.directed,
        source="_s",
        target="_t",
    )

    return res


def sort_arcs(streets, arcid):
    """
    Returns the dataframe with rows in source to target order for a unique id
    """

    streets["__arcid"] = _topological_arc_pos(streets, arcid)
    streets = streets.sort_values("__arcid").drop(columns="__arcid")
    return npd.set_network(streets, directed=False, edge="edge")


def _topological_index(successors: dict, start: int, ix: int):
    """
    returns list of new positions of edges topologically sorted
    """
    # single edge arc
    if len(successors.keys()) == 1:
        return [list(successors.values())[0][0]]

    pos, target = successors.pop(start)
    res = [pos]
    while target in successors.keys():
        pos, target = successors.pop(target)
        res.append(pos)
    return res


def _start_edge_of_arc(ndf, arcid):
    """
    return a Series with arcid as index, source of first node
    if arc is a loop, source is the first row connected to another edge
    else source is the source not connected to another target
    """

    # prepare result Series
    res = ndf[arcid].drop_duplicates().to_frame(arcid)
    res["start"] = pd.NA
    res = res.set_index(arcid)["start"]

    nodes = ndf.net.node_column_degree_inout(arcid)

    # start node if not loop
    st_nodes = nodes.loc[nodes.target == 0, "source"]
    st_nodes = st_nodes.reset_index(level=1)[NODE_ID_NAME]

    res.loc[st_nodes.index] = st_nodes

    # start node if simple loop
    cross_nodes = ndf.net.degree()
    cross_nodes = cross_nodes.loc[cross_nodes != 2]

    loops = ndf.loc[
        (ndf[arcid].isin(res.loc[res.isna()].index))
        & (ndf.net.sources.isin(cross_nodes.index))
    ]
    loops = loops.drop_duplicates(arcid).set_index(arcid)
    res.loc[loops.index] = loops.net.sources

    return res


def _topological_arc_pos(ndf, arcid, source=None, target=None):
    """
    returns re-ordered ndf from source to target in a single arc
    arcs must not contain complex self-loops
    """
    if source is None or target is None:
        source = "_s"
        target = "_t"
        df = ndf.net.expand_edges(source, target)
    else:
        df = ndf.copy()

    df["_simpid"] = no_self_loop_index(df, arcid)

    # dictionary of successors : arcid: {source:tuple of index, target}
    d = {arcid: {} for arcid in df["_simpid"].drop_duplicates()}
    for ix, i, s, t in df[["_simpid", "_s", "_t"]].itertuples(index=True, name=None):
        d[i][s] = (ix, t)

    starts = _start_edge_of_arc(df, "_simpid")

    res = []
    for ix, start in starts.iteritems():
        res.extend(_topological_index(d[ix], start, ix))

    res = pd.Series(data=res).sort_values()
    return pd.Series(data=res.index, index=res.values)


def no_self_loop_index(streets, refid, source=None, target=None):
    """
    Create a Series with new id to split self loop arcs sharing an refid
    to straight arcs and simple loop arc
    """
    if source is None or target is None:
        source = "_s"
        target = "_t"
        df = streets.net.expand_edges(source, target)
    else:
        df = streets.copy()

    nodes = streets.net.self_loops(refid, min_num=2)

    loops = df.loc[df[refid].isin(nodes.index)]

    # create a dictionary based graph, osm:{source:[(index, target]}
    G = {}
    rec = loops[[refid, source, "_t"]].to_records()

    for index, osmid, edge_s, edge_t in rec:
        if osmid not in G:
            G[osmid] = {}
        if edge_s not in G[osmid]:
            G[osmid][edge_s] = []
        G[osmid][edge_s].append((index, edge_t))

    # create newids of self loops
    new_ids = {}
    i = 0
    for osmid, group in loops.groupby(refid):

        loop_nodes = set(nodes.loc[nodes.index == osmid].to_list())
        targets = set(group["_t"].to_list())
        start_nodes = [
            k for k in G[osmid].keys() if k not in targets or k in loop_nodes
        ]

        for edge_s in start_nodes:
            for index, edge_t in G[osmid][edge_s]:
                i = i - 1
                new_ids[index] = i
                node = edge_t

                while (
                    node in G[osmid]
                    and len(G[osmid][node]) == 1
                    and node not in loop_nodes
                ):
                    index, node = G[osmid][node][0]
                    new_ids[index] = i

    new_ids = pd.Series(new_ids, dtype=int)

    res = streets[refid].copy()
    res.loc[new_ids.index] = new_ids

    return res
