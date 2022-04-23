# -*- coding: utf-8 -*-
import pandas as pd
from functools import singledispatchmethod


def dispatch_on_value(func):
    """
    Value-dispatch function decorator.
    Transforms a function into a value-dispatch function,
    which can have different behaviors based on the value of the first argument.
    """

    registry = {}

    def dispatch(value):
        try:
            return registry[value]
        except KeyError:
            return func

    def register(value, func=None):

        if func is None:
            return lambda f: register(value, f)

        registry[value] = func

        return func

    def wrapper(*args, **kw):
        return dispatch(args[1])(*args, **kw)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry

    return wrapper


class Backends:
    """
    Mixin with backend functions
    """

    # ------------------------------
    # dispatching
    # ------------------------------

    def backends(self):
        """
        return list of available backends
        """

        return None

    # ------------------------------
    # graph creation
    # ------------------------------

    @dispatch_on_value
    def graph(self, backend, attributes=None):
        raise NotImplementedError("Backend is not implemented")

    # ------------------------------
    # edges extraction
    # ------------------------------

    def _align_edges(self, edges):
        """
        align edges on self index
        """
        n = self.net.name
        return pd.merge(self.edges, edges, on=n, how="left").drop(columns=n)

    @singledispatchmethod
    def extract_edges(self, graph, attributes=None):
        raise NotImplementedError("backend is not implemented")

    # ------------------------------
    # nodes extraction
    # ------------------------------

    @singledispatchmethod
    def extract_nodes(self, graph, method, attributes=None):
        raise NotImplementedError("backend is not implemented")
