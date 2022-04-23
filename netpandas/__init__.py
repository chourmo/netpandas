"""AAdd graph functionnalities to a Pandas dataframe through its extension capabilities."""

# Add imports here
from .netpandas import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
