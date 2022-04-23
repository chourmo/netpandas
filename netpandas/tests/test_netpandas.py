"""
Unit and regression test for the netpandas package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import netpandas


def test_netpandas_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "netpandas" in sys.modules
