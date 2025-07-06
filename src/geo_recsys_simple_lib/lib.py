# src/geo_recsys_simple_lib/lib.py
import joblib
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from huggingface_hub import hf_hub_download
from .types import GeoCoordinates, GeoSearchQuery, GeoSearchResult, GeoSearchResults


class GeoNearestNeighbors(NearestNeighbors, TransformerMixin):
    """
    Geographic nearest neighbors search using scikit-learn's Haversine metric.
    
    This class extends scikit-learn's NearestNeighbors to provide specialized
    functionality for geographic route recommendation. It uses the Haversine
    formula to calculate distances between geographic coordinates on Earth's
    surface, making it suitable for location-based recommendation systems.
    
    Key Features:
    - Haversine distance calculation for accurate geographic distances
    - Radius-based search for finding routes within specified distance
    - Data validation using custom GeoCoordinates type
    - Model persistence with joblib and Hugging Face Hub integration
    - Structured result formatting with detailed route information
    
    Attributes:
        DESCRIPTION_COL (str): Column name for route descriptions
        START_LAT_COL (str): Column name for starting latitude coordinates
        START_LON_COL (str): Column name for starting longitude coordinates
        END_LAT_COL (str): Column name for ending latitude coordinates
        END_LON_COL (str): Column name for ending longitude coordinates
        EARTH_RADIUS_METERS (float): Earth's radius in meters for distance calculations
    """
    
    # Fixed column names for consistent data structure
    # These constants ensure that the model expects a standardized DataFrame format
    DESCRIPTION_COL = 'description'
    START_LAT_COL = 'start_latitude'
    START_LON_COL = 'start_longitude'
    END_LAT_COL = 'end_latitude'
    END_LON_COL = 'end_longitude'
    
    # Earth's radius in meters - used for converting radians to meters
    EARTH_RADIUS_METERS = 6371000.0

    def __init__(
        self,
        metric: str = 'haversine',
        algorithm: str = 'ball_tree',
        leaf_size: int = 30,
        n_jobs: int = -1
    ):
        """
        Initialize the GeoNearestNeighbors model.
        
        Args:
            metric: Distance metric to use
            algorithm: Algorithm for nearest neighbors search
            leaf_size: Leaf size for tree algorithms
            n_jobs: Number of parallel jobs (-1 uses all available processors)
        
        Note:
            The haversine metric requires coordinates in radians, which is handled
            automatically by the transform method.
        """
        super().__init__(
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs
        )

    def fit(self, X: pd.DataFrame) -> "GeoNearestNeighbors":
        """
        Fit the model with route data.
        
        Args:
            X: Route data containing required columns:
                - description: Route description text
                - start_latitude: Starting latitude in degrees (-90 to 90)
                - start_longitude: Starting longitude in degrees (-180 to 180)
                - end_latitude: Ending latitude in degrees (-90 to 90)
                - end_longitude: Ending longitude in degrees (-180 to 180)
        
        Returns:
            GeoNearestNeighbors: Self for method chaining
        """
        # Validate required columns exist in the DataFrame
        required_cols = [
            self.DESCRIPTION_COL, self.START_LAT_COL, self.START_LON_COL,
            self.END_LAT_COL, self.END_LON_COL
        ]
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate coordinates using GeoCoordinates type validation
        # This catches invalid latitude/longitude values early
        for idx, row in X.iterrows():
            try:
                # Validate starting coordinates
                GeoCoordinates(latitude=row[self.START_LAT_COL], longitude=row[self.START_LON_COL])
                # Validate ending coordinates
                GeoCoordinates(latitude=row[self.END_LAT_COL], longitude=row[self.END_LON_COL])
            except Exception as e:
                raise ValueError(f"Invalid coordinates at row {idx}: {e}")
        
        # Store the validated data for later use in result formatting
        # Only keep required columns to reduce memory usage
        self.data_ = X[required_cols].copy()
        
        # Create coordinate array for the nearest neighbors algorithm
        # We use starting coordinates as the primary search points
        coords = np.column_stack((
            X[self.START_LAT_COL].astype(np.float64),
            X[self.START_LON_COL].astype(np.float64)
        ))
        
        # Transform coordinates from degrees to radians
        # Haversine metric requires radians for proper distance calculation
        coords_rad = self.transform(coords)
        
        # Fit the underlying scikit-learn NearestNeighbors model
        super().fit(coords_rad)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from degrees to radians.
        
        The haversine metric in scikit-learn requires coordinates in radians
        rather than degrees. This method handles the conversion automatically.
        
        Args:
            X: Coordinate array with shape (n_samples, 2) 
               where columns are [latitude, longitude] in degrees
        
        Returns:
            np.ndarray: Coordinates converted to radians with same shape
        """
        X = np.asarray(X)
        # Handle single coordinate pair by reshaping to 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Convert degrees to radians using numpy's radians function
        return np.radians(X)

    def predict(self, X: GeoSearchQuery) -> GeoSearchResults:
        """
        Search for routes within geographic radius.
        
        Args:
            X: Search query containing:
                - coordinates: GeoCoordinates object with search center
                - radius_meters: Search radius in meters
        
        Returns:
            GeoSearchResults: Structured results containing:
                - results: List of GeoSearchResult objects
                - total_found: Number of routes found
        """
        # Ensure the model has been fitted before attempting predictions
        check_is_fitted(self, ['data_'])
        
        # Extract coordinates from the search query and convert to numpy array
        coords_array = X.coordinates.to_array()
        # Transform coordinates to radians for haversine calculation
        coords_rad = self.transform(coords_array)
        
        # Convert search radius from meters to radians
        # This is necessary because haversine metric works in radians
        radius_rad = X.radius_meters / self.EARTH_RADIUS_METERS
        
        # Find all neighbors within the specified radius
        # Returns distances in radians and indices of matching routes
        distances_rad, indices = self.radius_neighbors(
            coords_rad, radius=radius_rad, return_distance=True
        )
        
        # Convert distances back to meters for user-friendly results
        distances_m = distances_rad[0] * self.EARTH_RADIUS_METERS
        # Combine indices and distances for sorting
        results = list(zip(indices[0], distances_m))
        # Sort results by distance (closest first)
        results.sort(key=lambda x: x[1])
        
        # Format results with full route information
        return self._format_results(results)

    def _format_results(self, results: List[tuple]) -> GeoSearchResults:
        """
        Format search results with route details and validation.
        
        Args:
            results: List of (index, distance) tuples from search
        
        Returns:
            GeoSearchResults: Structured results with validated route data
            
        Note:
            This method ensures all returned coordinates are valid by creating
            GeoCoordinates objects, which perform automatic validation.
        """
        formatted_results = []
        
        for idx, dist in results:
            # Get the route data for this result
            row = self.data_.iloc[idx]
            
            # Create a structured result object with validated coordinates
            result = GeoSearchResult(
                index=int(idx),  # Original DataFrame index
                description=str(row[self.DESCRIPTION_COL]),  # Route description
                # Validate and format starting coordinates
                start_coordinates=GeoCoordinates(
                    latitude=float(row[self.START_LAT_COL]),
                    longitude=float(row[self.START_LON_COL])
                ),
                # Validate and format ending coordinates
                end_coordinates=GeoCoordinates(
                    latitude=float(row[self.END_LAT_COL]),
                    longitude=float(row[self.END_LON_COL])
                ),
                # Round distance to 2 decimal places for readability
                geo_distance=float(round(dist, 2))
            )
            formatted_results.append(result)
        
        # Return structured results with metadata
        return GeoSearchResults(
            results=formatted_results,
            total_found=len(formatted_results)
        )

    def __call__(self, X: GeoSearchQuery) -> GeoSearchResults:
        """
        Make the model callable - alias for predict method.
        
        This allows the model to be used as a function, providing a more
        intuitive interface: model(query) instead of model.predict(query).
        
        Args:
            X: Search query object
        
        Returns:
            GeoSearchResults: Search results with validated route data
        """
        return self.predict(X)

    def save_model(self, filepath: str = "models/geo_nearest_neighbors.joblib") -> None:
        """
        Save complete model state to disk.
        
        This method serializes the entire model including:
        - Fitted parameters and data structures
        - Training data for result formatting
        - Model configuration and hyperparameters
        
        Args:
            filepath (str): Path where to save the model file
            
        Note:
            Uses joblib compression level 3 for good balance between
            file size and save/load speed.
        """
        try:
            # Verify model is fitted before saving
            check_is_fitted(self, ['data_'])
        except NotFittedError:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        # Save with compression to reduce file size
        joblib.dump(self, filepath, compress=3)

    def load_model(self, filepath: str = "models/geo_nearest_neighbors.joblib") -> "GeoNearestNeighbors":
        """
        Load model state into current instance.
        
        This method loads a previously saved model and updates the current
        instance with all saved parameters and data.
        
        Args:
            filepath (str): Path to the saved model file
        
        Returns:
            GeoNearestNeighbors: Self for method chaining
            
        Note:
            This method modifies the current instance rather than creating
            a new one. Use from_file() for creating a new instance.
        """
        # Load the saved model
        loaded_model = joblib.load(filepath)
        # Update current instance with loaded model's state
        self.__dict__.update(loaded_model.__dict__)
        return self

    @classmethod
    def from_file(cls, filepath: str = "models/geo_nearest_neighbors.joblib") -> "GeoNearestNeighbors":
        """
        Create new instance from saved file.
        
        This class method creates a completely new GeoNearestNeighbors instance
        from a previously saved model file.
        
        Args:
            filepath: Path to the saved model file
        
        Returns:
            GeoNearestNeighbors: New instance loaded from file
        """
        return joblib.load(filepath)

    @classmethod
    def from_huggingface_hub(
        cls, 
        repo_id: str,
        filename: str = "models/geo_nearest_neighbors.joblib",
        cache_dir: str = "models/",
        force_download: bool = False,
        token: str = None
    ) -> "GeoNearestNeighbors":
        """
        Download and load model from Hugging Face Hub.
        
        Args:
            repo_id (str): Repository ID on Hugging Face Hub
                           Format: "username/model-name" or "organization/model-name"
            filename (str): Name of the model file in the repository
                            Should match the filename used when uploading
            cache_dir (str): Local directory to cache downloaded files
                             Avoids re-downloading on subsequent calls
            force_download (bool): Whether to force re-download even if cached
                                   Useful for getting updated model versions
            token (str): Hugging Face authentication token
                         Required for private repositories, optional for public
        
        Returns:
            GeoNearestNeighbors: New instance loaded from Hugging Face Hub
        """

        try:
            # Download model file from Hub
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token
            )
            
            # Load and return the model
            return joblib.load(model_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to download model from Hugging Face Hub: {e}")