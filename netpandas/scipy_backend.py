# -*- coding: utf-8 -*-

import numpy as np

from . import backends

try:
    import scipy.sparse as ss

    SCIPY_GRAPH = ss.coo.coo_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


SCIPY_NAME = "scipy"


class Scipy_backend(backends.Backends):
    """
    Scipy backend
    """

    @backends.Backends.graph.register(SCIPY_NAME)
    def scipy_graph(self, backend, attributes=None):
        """
        create a scipy coo graph
        """

        # raise error if self_obj is not geodataframe

        if type(attributes) is list and len(attributes) > 1:
            raise ValueError("scipy graph does not supports multiple attributes")

        rows = self.sources.to_numpy()
        targets = self.targets.to_numpy()

        s = max(rows.max(), targets.max()) + 1

        if attributes is None:
            data = np.ones(len(rows), np.uint32)
        else:
            data = self._obj[attributes[0]].to_numpy()

        return ss.coo_matrix((data, (rows, targets)), shape=(s, s))
