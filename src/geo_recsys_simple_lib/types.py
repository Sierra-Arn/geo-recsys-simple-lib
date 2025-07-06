# src/geo_recsys_simple_lib/types.py
import numpy as np
from typing import List
from pydantic import BaseModel, Field

class GeoCoordinates(BaseModel):
    """Geographic coordinates with validation."""
    
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [lat, lon]."""
        return np.array([self.latitude, self.longitude])

class GeoSearchQuery(BaseModel):
    """Geographic search query with coordinates and radius."""
    
    coordinates: GeoCoordinates
    radius_meters: float = Field(..., gt=0, description="Search radius in meters")

class GeoSearchResult(BaseModel):
    """Single search result with validation."""
    
    index: int = Field(..., ge=0, description="Index in original dataset")
    description: str = Field(..., min_length=1, description="Route description")
    start_coordinates: GeoCoordinates = Field(..., description="Start point coordinates")
    end_coordinates: GeoCoordinates = Field(..., description="End point coordinates")
    geo_distance: float = Field(..., ge=0, description="Distance in meters")

class GeoSearchResults(BaseModel):
    """Collection of search results with validation."""
    
    results: List[GeoSearchResult] = Field(..., description="List of found routes")
    total_found: int = Field(..., ge=0, description="Total number of results")
