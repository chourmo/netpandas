# -*- coding: utf-8 -*-

import pandas as pd
import netpandas as npd

# -------------------
# Paths utility functions, a path is a list of nodes
# -------------------


def is_list_dtype(df):
    """
    Check whether the provided Series is of a list dtype
    """

    return df.apply(lambda x: isinstance(x, list)).all()


def expand_paths(paths, ndf, path_col=None):
    """
    Expand paths of nodes to edges based on a net dataframe
    a path is either a pd.Series or pd.DataFrame or simple list if single path
    if Dataframe, the path data is in path_col

    return a pandas dataframe with on edge/path per row, paths index
    and data duplicated on each edge on a path
    if ndf is a geodataframe, keep spatial properties
    """

    if ndf.net.has_duplicated_edges():
        raise ValueError("ndf shall not have duplicated edges")

    # test if paths index is unique
    if isinstance(paths, list):
        df = pd.Series({0: paths}).explode().to_frame("_s")

    elif isinstance(paths, pd.Series):
        if not paths.index.is_unique:
            raise ValueError("Paths index must be unique")
        df = paths.explode().to_frame("_s")

    elif isinstance(paths, pd.DataFrame) and path_col in paths.columns:
        if not paths.index.is_unique:
            raise ValueError("Paths index must be unique")
        df = paths.explode(path_col).rename(columns={path_col: "_s"})

    else:
        raise AttributeError("paths is neither a list, a Series or a DataFrame")

    df["_s"] = df["_s"].astype("Int64")
    df.index.name = "edge_id"
    df = df.reset_index()

    df[["edge_id_n", "_t"]] = df[["edge_id", "_s"]].shift(-1)
    df = df.loc[df.edge_id == df.edge_id_n]
    df["_t"] = df["_t"].astype("Int64")

    del df["edge_id_n"]

    df = npd.set_network(
        df, directed=True, source="_s", target="_t", edge_name=ndf.net.name
    )
    
    df = pd.merge(df, ndf, on=ndf.net.name, how="left").set_index("edge_id")
    df.index.name = paths.index.name

    try:
        df = df.set_geometry(ndf.geometry.name, ndf.crs)
    except:
        pass

    return df


def paths_distance(paths, ndf, weight):
    """
    find distance on paths, based on ndf netdataframe and weight column
    returns a Series with same index as paths
    """
    df = expand_paths(paths, ndf)[[weight]]
    df = df.groupby(level=0)[weight].sum()

    return df


def make_path(predecessors, source, target):
    """return path list from a predecessor list"""

    res = [target]
    n = target

    while n != source:
        n = predecessors[n]
        res.append(n)

    # revert order from to source to target
    res.reverse()
    return res


def add_paths(df1, df2):
    """
    append df2 Series to df1 Series in a list Series
    """

    res = pd.concat([df1.explode(), df2.explode()])
    return res.groupby(level=0).agg(list)
