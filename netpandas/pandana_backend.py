# -*- coding: utf-8 -*-

from . import backends

try:
    import pandana as pdn

    PANDANA_GRAPH = pdn.network.Network
    HAS_PANDANA = True
except ImportError:
    HAS_PANDANA = False


PANDANA_NAME = "pandana"


class Pandana_backend(backends.Backends):
    """
    Pandana backend
    """

    @backends.Backends.graph.register(PANDANA_NAME)
    def pandana_graph(self, backend, attributes):
        """
        return a pandana graph, if pandana is not installed, return None
        attributes is a list of column names to add as attributes
        """

        if not self.is_spatial:
            raise ValueError("The dataframe must be spatial to use pandana")

        # generate nodes with coordinates

        geodf = self.node_geometry()

        if attributes is None:
            raise ValueError("Attributes must be set for pandana graphs")

        twoway = not self.directed

        # make pandana graph
        graph = pdn.network.Network(
            geodf.x,
            geodf.y,
            self.sources,
            self.targets,
            edge_weights=self._obj[attributes],
            twoway=twoway,
        )

        return graph
