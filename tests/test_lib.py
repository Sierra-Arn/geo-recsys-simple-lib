# tests/test_lib.py
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from src.geo_recsys_simple_lib import GeoNearestNeighbors, GeoCoordinates, GeoSearchQuery, GeoSearchResult, GeoSearchResults
from sklearn.exceptions import NotFittedError


class TestGeoNearestNeighbors:
    """Test suite for GeoNearestNeighbors class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample route data for testing."""
        return pd.DataFrame({
            'description': ['Route A', 'Route B', 'Route C', 'Route D'],
            'start_latitude': [40.7128, 34.0522, 41.8781, 29.7604],
            'start_longitude': [-74.0060, -118.2437, -87.6298, -95.3698],
            'end_latitude': [40.7589, 34.0928, 41.8881, 29.8104],
            'end_longitude': [-73.9851, -118.2437, -87.6198, -95.3598]
        })
    
    @pytest.fixture
    def invalid_data(self):
        """Create invalid data for testing error cases."""
        return pd.DataFrame({
            'description': ['Route A'],
            'start_latitude': [91.0],  # Invalid latitude
            'start_longitude': [-74.0060],
            'end_latitude': [40.7589],
            'end_longitude': [-73.9851]
        })
    
    @pytest.fixture
    def incomplete_data(self):
        """Create incomplete data missing required columns."""
        return pd.DataFrame({
            'description': ['Route A'],
            'start_latitude': [40.7128],
            # Missing other required columns
        })
    
    @pytest.fixture
    def geo_model(self):
        """Create a GeoNearestNeighbors instance."""
        return GeoNearestNeighbors()
    
    @pytest.fixture
    def fitted_model(self, geo_model: GeoNearestNeighbors, sample_data: pd.DataFrame):
        """Create a fitted GeoNearestNeighbors instance."""
        return geo_model.fit(sample_data)
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        model = GeoNearestNeighbors()
        params = model.get_params()
        
        assert params['metric'] == 'haversine'
        assert params['algorithm'] == 'ball_tree'
        assert params['leaf_size'] == 30
        assert params['n_jobs'] == -1
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = GeoNearestNeighbors(
            metric='euclidean',
            algorithm='kd_tree',
            leaf_size=50,
            n_jobs=1
        )
        params = model.get_params()
        
        assert params['metric'] == 'euclidean'
        assert params['algorithm'] == 'kd_tree'
        assert params['leaf_size'] == 50
        assert params['n_jobs'] == 1
    
    def test_fit_valid_data(self, geo_model: GeoNearestNeighbors, sample_data: pd.DataFrame):
        """Test fitting with valid data."""
        result = geo_model.fit(sample_data)
        
        # Should return self
        assert result is geo_model
        
        # Should store data
        assert hasattr(geo_model, 'data_')
        assert len(geo_model.data_) == len(sample_data)
        
        # Should have required columns
        expected_cols = [
            'description', 'start_latitude', 'start_longitude',
            'end_latitude', 'end_longitude'
        ]
        assert list(geo_model.data_.columns) == expected_cols
    
    def test_fit_missing_columns(self, geo_model: GeoNearestNeighbors, incomplete_data: pd.DataFrame):
        """Test fitting with missing required columns."""
        with pytest.raises(ValueError, match="Missing required columns"):
            geo_model.fit(incomplete_data)
    
    def test_fit_invalid_coordinates(self, geo_model: GeoNearestNeighbors, invalid_data: pd.DataFrame):
        """Test fitting with invalid coordinates."""
        with pytest.raises(ValueError, match="Invalid coordinates"):
            geo_model.fit(invalid_data)
    
    def test_transform_single_coordinate(self, geo_model: GeoNearestNeighbors):
        """Test transforming single coordinate pair."""
        coords = np.array([40.7128, -74.0060])
        result = geo_model.transform(coords)
        
        expected = np.radians(coords).reshape(1, -1)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_transform_multiple_coordinates(self, geo_model: GeoNearestNeighbors):
        """Test transforming multiple coordinate pairs."""
        coords = np.array([
            [40.7128, -74.0060],
            [34.0522, -118.2437]
        ])
        result = geo_model.transform(coords)
        
        expected = np.radians(coords)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_predict_not_fitted(self, geo_model: GeoNearestNeighbors):
        """Test prediction on unfitted model."""
        query = GeoSearchQuery(
            coordinates=GeoCoordinates(latitude=40.7128, longitude=-74.0060),
            radius_meters=1000
        )
        
        with pytest.raises(NotFittedError):
            geo_model.predict(query)
    
    def test_predict_valid_query(self, fitted_model: GeoNearestNeighbors):
        """Test prediction with valid query."""
        query = GeoSearchQuery(
            coordinates=GeoCoordinates(latitude=40.7128, longitude=-74.0060),
            radius_meters=10000  # 10km radius
        )
        
        results = fitted_model.predict(query)
        
        # Should return GeoSearchResults
        assert isinstance(results, GeoSearchResults)
        
        # Should have total_found attribute
        assert hasattr(results, 'total_found')
        assert results.total_found >= 0
        
        # Each result should be GeoSearchResult
        for result in results.results:
            assert isinstance(result, GeoSearchResult)
            assert hasattr(result, 'index')
            assert hasattr(result, 'description')
            assert hasattr(result, 'start_coordinates')
            assert hasattr(result, 'end_coordinates')
            assert hasattr(result, 'geo_distance')
            
            # Check coordinate types
            assert isinstance(result.start_coordinates, GeoCoordinates)
            assert isinstance(result.end_coordinates, GeoCoordinates)

    
    def test_predict_results_sorted_by_distance(self, fitted_model: GeoNearestNeighbors):
        """Test that prediction results are sorted by distance."""
        query = GeoSearchQuery(
            coordinates=GeoCoordinates(latitude=40.7128, longitude=-74.0060),
            radius_meters=50000  # Large radius to get multiple results
        )
        
        results = fitted_model.predict(query)
        
        if len(results.results) > 1:
            distances = [result.geo_distance for result in results.results]
            assert distances == sorted(distances)

    def test_format_results(self, fitted_model: GeoNearestNeighbors):
        """Test result formatting."""
        # Mock results data
        results = [(0, 1234.56), (1, 2345.67)]
        
        formatted = fitted_model._format_results(results)
        
        # Should return GeoSearchResults
        assert isinstance(formatted, GeoSearchResults)
        assert formatted.total_found == 2
        
        # Check first result
        first_result = formatted.results[0]
        assert first_result.index == 0
        assert isinstance(first_result.description, str)
        assert isinstance(first_result.start_coordinates, GeoCoordinates)
        assert isinstance(first_result.end_coordinates, GeoCoordinates)
        assert first_result.geo_distance == 1234.56
    
    def test_call_method(self, fitted_model: GeoNearestNeighbors):
        """Test that model is callable and works same as predict."""
        query = GeoSearchQuery(
            coordinates=GeoCoordinates(latitude=40.7128, longitude=-74.0060),
            radius_meters=10000
        )
        
        # Call using __call__ method
        call_results = fitted_model(query)
        
        # Call using predict method
        predict_results = fitted_model.predict(query)
        
        assert isinstance(call_results, GeoSearchResults)
        assert isinstance(predict_results, GeoSearchResults)
        
        # Should return same results
        assert call_results.total_found == predict_results.total_found
        assert len(call_results.results) == len(predict_results.results)
        
        # Compare individual results
        for call_result, predict_result in zip(call_results.results, predict_results.results):
            assert call_result.index == predict_result.index
            assert call_result.description == predict_result.description
            assert call_result.geo_distance == predict_result.geo_distance
            assert call_result.start_coordinates.latitude == predict_result.start_coordinates.latitude
            assert call_result.start_coordinates.longitude == predict_result.start_coordinates.longitude
            assert call_result.end_coordinates.latitude == predict_result.end_coordinates.latitude
            assert call_result.end_coordinates.longitude == predict_result.end_coordinates.longitude

    def test_save_model_not_fitted(self, geo_model: GeoNearestNeighbors):
        """Test saving unfitted model."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError, match="Model must be fitted before saving"):
                geo_model.save_model(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_save_and_load_model(self, fitted_model: GeoNearestNeighbors):
        """Test saving and loading fitted model."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name
            
            try:
                # Save model
                fitted_model.save_model(tmp_path)
                assert os.path.exists(tmp_path)
                
                # Load model using from_file (нет метода load_model)
                loaded_model = GeoNearestNeighbors()
                loaded_model.load_model(tmp_path)
                
                # Should have same data
                pd.testing.assert_frame_equal(loaded_model.data_, fitted_model.data_)
                
                # Should be able to predict
                query = GeoSearchQuery(
                    coordinates=GeoCoordinates(latitude=40.7128, longitude=-74.0060),
                    radius_meters=10000
                )
                results = loaded_model.predict(query)
                assert isinstance(results, GeoSearchResults)
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def test_from_file_class_method(self, fitted_model: GeoNearestNeighbors):
        """Test loading model using class method."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save model
            fitted_model.save_model(tmp_path)
            
            # Load using class method
            loaded_model = GeoNearestNeighbors.from_file(tmp_path)
            
            # Should be GeoNearestNeighbors instance
            assert isinstance(loaded_model, GeoNearestNeighbors)
            
            # Should have same data
            pd.testing.assert_frame_equal(loaded_model.data_, fitted_model.data_)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
