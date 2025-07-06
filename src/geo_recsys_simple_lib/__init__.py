"""Simple Geographic Recomendation System Library."""

__version__ = "0.1.0"

from .lib import GeoNearestNeighbors
from .types import GeoCoordinates, GeoSearchQuery, GeoSearchResult, GeoSearchResults

__all__ = [
    "GeoNearestNeighbors",
    "GeoCoordinates",
    "GeoSearchQuery",
    "GeoSearchResult",
    "GeoSearchResults"
]