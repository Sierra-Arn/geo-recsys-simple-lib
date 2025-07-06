# **Simple Geographic Recommendation System Library**

> **Note:** This is a simplified implementation built with **scikit-learn** and designed for learning and experimentation. For production deployments, consider additional features like advanced scoring algorithms and scalability optimizations.

## **Key Features**

- **Geographic Proximity Search**:  
Find routes within specified radius using Haversine distance calculation.
- **Type-Safe Input Validation**:  
Pydantic models ensure data integrity and provide clear error messages.
- **Scikit-learn Compatible**:  
Implements standard fit/predict/transform interface with TransformerMixin.
- **High Performance**:  
Optimized with scikit-learn's BallTree algorithm for fast geographic queries.
- **Clean Architecture**:  
Simplified codebase focused on core geographic functionality.
- **Serializable**:  
Save and load trained models with joblib.

## **Project Structure**

```bash
geo-recsys-simple-lib/
├── dist/
│   ├── geo_recsys_simple_lib-0.1.0-py3-none-any.whl    # Built wheel package
│   └── geo_recsys_simple_lib-0.1.0.tar.gz              # Built source distribution
├── src/
│   └── geo_recsys_model_lib/
│       ├── __init__.py                                 # Package initialization and exports
│       ├── lib.py                                      # Geographic recommendation system
│       └── types.py                                    # Custom types
├── tests/
│   ├── __init__.py                                     # Test package initialization
│   └── test_lib.py                                     # Geographic recommendation system tests
├── pixi.toml                                           # Pixi project configuration and dependencies
├── pixi.lock                                           # Locked dependency versions
├── setup.py                                            # Python package build configuration
├── LICENSE                                             # BSD-3-Clause license
├── README.md                                           # Project documentation
└── THIRD-PARTY-LICENSES.md                             # Third-party dependency licenses
```

## **Installation**

```bash
pip install https://github.com/sierra-arn/geo-recsys-simple-lib/releases/latest/download/geo_recsys_simple_lib-0.1.0-py3-none-any.whl
```

## **Quick Start**

```python
import pandas as pd
from geo_recsys_model_lib import GeoNearestNeighbors, GeoSearchQuery, GeoCoordinates

# Prepare your route data
data = pd.DataFrame({
    'description': ['Mountain hiking trail', 'City bike route', 'Coastal walk'],
    'start_latitude': [40.7128, 40.7589, 40.6892],
    'start_longitude': [-74.0060, -73.9851, -74.0445],
    'end_latitude': [40.7300, 40.7505, 40.7000],
    'end_longitude': [-74.0200, -73.9733, -74.0350]
})

# Initialize and train the model
model = GeoNearestNeighbors()
model.fit(data)

# Save trained model
model.save_model("models/geo_model.joblib")

# Load trained model
model = GeoNearestNeighbors.from_file("models/geo_model.joblib")

# Create type-safe search query
coordinates = GeoCoordinates(latitude=40.7128, longitude=-74.0060)
query = GeoSearchQuery(coordinates=coordinates, radius_meters=5000)

# Make predictions
results = model.predict(query)
print(results)
```

## **API Reference**

>**Note:** See [types](src/geo_recsys_simple_lib/types.py) for complete type annotations.

### **I. Input Format**

```python
class GeoCoordinates(BaseModel):
    latitude: float
    longitude: float

class GeoSearchQuery(BaseModel):
    coordinates: GeoCoordinates
    radius_meters: float
```

### **II. Output Format**
```python
class GeoSearchResult(BaseModel):
    index: int
    description: str
    start_coordinates: GeoCoordinates
    end_coordinates: GeoCoordinates
    geo_distance: float

class GeoSearchResults(BaseModel):
    results: List[GeoSearchResult]
    total_found: int
```

## **Development and Testing**

This project uses **Pixi** for dependency management and task automation.

### **Setup Development Environment**
```bash
git clone https://github.com/siearra-arn/geo-recsys-simple-lib.git
cd geo-recsys-simple-lib
pixi install --all
```

### **Available Pixi Tasks**

1. **Testing**  
Runs the full test suite with coverage reporting and suppresses deprecation warnings.

    ```bash
    pixi run test
    ```

2. **Building**  
Creates distribution packages (wheel and source) for PyPI publishing.

    ```bash
    pixi run build
    ```

### **Environment Features**
- **`test`**: Includes pytest and coverage tools
- **`dev`**: Includes build tools for package creation  
- **`default`**: Core runtime dependencies only

## **License**

This project is distributed under the [BSD-3-Clause License](LICENSE).

> **Third-Party Dependencies**  
> This project integrates multiple open-source libraries with separate licenses. Review [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES.md) for complete dependency licensing information.
