# -*- coding: utf-8 -*-

import pandas as pd
import netpandas as npd
import shapely as sh

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


try:
    import geopandas as gpd

    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False

GEOMETRY_NAME = "geometry"
NODE_ID_NAME = "nodeid"


class SpatialFunctions(object):
    """
    Mixin of spatial functions added to NetworkAccessor
    """

    @property
    def is_spatial(self):
        """
        test if a net dataframe is spatial and if geopandas is installed
        """

        if not HAS_SPATIAL:
            return False
        else:
            return isinstance(self._obj, gpd.geodataframe.GeoDataFrame)

    def spatial_dataframe(self, geom_types=None):
        """
        check if obj is a geodataframe and as geometry types only in certain types
        geom_types is 'point', 'line', 'polygon' or None, then do not check types
        """

        b = self.is_spatial()
        if ~b or geom_types is None:
            return b

        if geom_types == "point":
            gtypes = set(["Point", "MultiPoint"])
        elif geom_types == "line":
            gtypes = set(["LineString", "MultiLineString"])
        elif geom_types == "polygon":
            gtypes = set(["Polygon", "MuliPolygon"])
        else:
            raise ValueError("geom_types not equal to point, line or polygon")

        type_list = self._obj.geometry.geom_type.drop_duplicates().to_list()

        return set(type_list).issubset(gtypes)

    # --------------------
    # node functions
    # --------------------

    def node_geometry(self):
        """
        return a GeoSeries of point geometries indexed by node_id
        """

        df = self.expand_edges(source="_s", target="_t")

        # source points
        df_s = df.drop_duplicates("_s").set_index("_s")
        points = df_s.geometry.interpolate(0, normalized=True)

        # target points
        df = df.loc[~df["_t"].isin(points.index.values)]
        df = df.drop_duplicates("_t").set_index("_t")

        points = pd.concat([points, df.geometry.interpolate(1, normalized=True)])

        return points.sort_index()

    def nodes_at_distance(self, geoms, distance, sort=True, closest=False):
        """
        return node ids at distance of GeoSeries
        optionnaly sort by index and distance

        returns:
            a GeoDataframe with same index values as geoms,
               repeated if multiply results, pd.Na if no edge at distance
            a nodeid column corresponding to node ids
            a distance column
        """

        df_g = geoms.copy()
        if isinstance(df_g, gpd.geoseries.GeoSeries):
            df_g = gpd.GeoDataFrame(geometry=df_g)

        df_g = df_g.rename_geometry("ref_geom")

        # make rectangular buffer
        df_g["buffer"] = df_g.buffer(distance, resolution=1)
        df_g = df_g.set_geometry("buffer", crs=geoms.crs)

        nodes = gpd.GeoDataFrame(geometry=self.node_geometry())

        res = gpd.sjoin(df_g, nodes, predicate="contains", how="left")
        res = res.rename(columns={"index_right": NODE_ID_NAME})

        res = pd.merge(res, nodes, left_on=NODE_ID_NAME, right_index=True, how="left")

        res["distance"] = res["geometry"].distance(res["ref_geom"])

        # limit distance
        res = res.loc[(res["distance"] <= distance) | (res["distance"].isna())]

        if closest:
            res.index.name = "index"
            res = res.sort_values(["index", "distance"])
            res = res.groupby(["index"]).first()
            res.index.name = geoms.index.name
            return res[[NODE_ID_NAME, "distance"]]

        elif sort:
            res.index.name = "index"
            res = res.sort_values(["index", "distance"])
            res.index.name = geoms.index.name
            return res[[NODE_ID_NAME, "distance"]]

        else:
            return res[[NODE_ID_NAME, "distance"]]

    def node_clusters(self, distance, size):
        """
        Find clusters of nodes

        args
            streets : streetspy dataframe
            distance : maximum distance between nodes
            size : minimum cluster size

        returns a Series with node id as index and cluster integer, drops nodes not in a cluster
        """

        # extract nodes x and y in projected space

        pts = self.node_geometry().to_frame("geometry")
        pts = pts.set_geometry("geometry")
        pts["x"] = pts.geometry.x
        pts["y"] = pts.geometry.y

        # create sparse matrix of nearest neighbours

        radius = distance * 5.0

        # use DBSCAN from sklearn

        neigh = NearestNeighbors(radius=radius)
        neigh.fit(pts[["x", "y"]].to_numpy())
        A = neigh.radius_neighbors_graph(pts[["x", "y"]].to_numpy())

        res = DBSCAN(eps=distance, metric="precomputed", min_samples=size).fit(A)

        pts["clusters"] = res.labels_

        return pts.loc[pts.clusters != -1]["clusters"]

    # ----------------------
    # edge functions
    # ----------------------

    def edges_at_distance(self, geoms, distance, k_nearest=None, edge_attrs=None):
        """
        find edges within distance of a GeoDataframe()

        Args :
            distance : value in geoms crs
            nearest : keep the k nearest, if None keep all values
            columns : columns of graph to keep in results,
                      None returns no columns, True all columns

        Returns:
            a GeoDataframe with same content as geoms
            geoms is repeated if multiple results, and dropped from result if no edge in distance
            edge_geom : the edge geometry
            source and target : source and target columns named as edges
            distance : distance between edge geometry and input geometry
            result is sorted by index and the distance between geoms and edges
        """

        if not geoms.index.is_unique:
            raise ValueError("geoms index must be unique")

        df_g = geoms.copy()

        # make rectangular buffer
        df_g["buffer"] = df_g.buffer(distance, resolution=1)
        df_g = df_g.set_geometry("buffer", crs=geoms.crs)

        edges = self._obj.copy()

        if edge_attrs is None:
            edges = edges[[edges.geometry.name, self.name]]
        elif isinstance(edge_attrs, list):
            cols = edge_attrs + [edges.geometry.name, self.name]
            edges = edges[cols]
        elif isinstance(edge_attrs, str):
            cols = [edge_attrs, edges.geometry.name, self.name]
            edges = edges[cols]

        edges = edges.to_crs(geoms.crs)
        edges["edge_geom"] = edges.geometry.copy()

        res = gpd.sjoin(df_g, edges, predicate="intersects", how="left")
        res = res.drop(columns=["index_right", "buffer"])

        # drop missing edges
        res = res.loc[~res[self.name].isna()]

        # distance between point and edge, limit to max distance
        res["distance"] = res[geoms.geometry.name].distance(res["edge_geom"])
        res = res.loc[(res["distance"] <= distance)]

        # sort by index and ascending distance
        res.index.name = "geom_index"

        if k_nearest is not None:
            res = res.sort_values(["geom_index", "distance"], ascending=True)
            res = res.groupby(level=res.index.name).head(k_nearest)

            # groupby reverts type of edge_geom to Series
            res = res.set_geometry("edge_geom", crs=geoms.crs)
            res = res.set_geometry(geoms.geometry.name)

        res.index.name = geoms.index.name

        return res


# -------------------------
# create network source and target from geodataframe
# -------------------------


def connect_geodataframe(gdf, distance):
    """
    gdf : geopandas dataframe of linestrings
    add from and to integer values
    -------
    return : a netDataFrame
    """

    # start nodes
    res = gpd.GeoDataFrame(geometry=gdf.geometry.interpolate(0, normalized=True))
    res = res.rename_geometry("_source_geom")
    res["_s"] = range(len(res))

    # end nodes
    res = res.assign(_target_geom=gdf.geometry.interpolate(1, normalized=True))
    res["_t"] = res["_s"] + len(res)

    nodes = (
        res[["_source_geom", "_s"]]
        .set_index("_s")
        .rename(columns={"_source_geom": "geometry"})
    )
    df = (
        res[["_target_geom", "_t"]]
        .set_index("_t")
        .rename(columns={"_target_geom": "geometry"})
    )
    nodes = pd.concat([nodes, df])
    nodes = nodes.set_geometry("geometry", crs=gdf.crs)

    res = res[["_s", "_t"]].copy()

    # find pairs in distance, resolution=1 to make rectangular buffer
    nodes["buffer"] = nodes.geometry.buffer(distance, resolution=1)

    # find pairs of stops in buffer, including self pairs
    pairs = gpd.sjoin(
        nodes[[nodes.geometry.name]],
        nodes.set_geometry("buffer")[["buffer"]],
        predicate="within",
    )

    pairs.index.name = "index_left"
    pairs = pairs.reset_index()

    pairs = npd.set_network(
        pairs, directed=True, source="index_left", target="index_right"
    )

    res[["_s", "_t"]] = pairs.net.edge_components()

    return npd.set_network(res, source="_s", target="_t", directed=True)


# -----------------------------------------------------
# other functions


def linemerge(gdf):
    geom = sh.multilinestrings(gdf.geometry.array, indices=gdf.index.to_numpy())
    geom = sh.line_merge(geom)
    index = gdf.index.drop_duplicates()
    return gpd.GeoSeries(data=geom, index=index, crs=gdf.crs)


def reverse(gdf):
    """
    return the geometries of a GeoSeries in reverse direction
    """
    geom = gdf.geometry.array
    return sh.reverse(geom)
