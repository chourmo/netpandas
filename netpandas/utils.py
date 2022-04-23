# -*- coding: utf-8 -*-


# -------------------
# Errors
# -------------------


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class NoNetworkError(Error):
    """Exception raised if no graph setup"""

    def __init__(self, message):
        self.message = "graph has not been set up, use set_edge_graph or set_node_graph"


class NotImportedError(Error):
    """Exception raised if no graph setup"""

    def __init__(self, message, library):
        self.message = "{0} has not been imported".format(library)
