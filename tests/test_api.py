"""
API Tests for ScoutIA Pro
Tests FastAPI endpoints and functionality.
"""

import pytest
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Only import if FastAPI is available
try:
    from fastapi.testclient import TestClient
    from backend.main import app
    
    client = TestClient(app)
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    client = None


@pytest.mark.skipif(not API_AVAILABLE, reason="FastAPI not available")
class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data
        assert "version" in data


@pytest.mark.skipif(not API_AVAILABLE, reason="FastAPI not available")
class TestPredictEndpoint:
    """Test prediction endpoint."""
    
    def test_predict_with_valid_data(self):
        """Test prediction with valid player data."""
        player_data = {
            "age": 25,
            "matches_played": 30,
            "minutes_played": 2400,
            "goals": 5,
            "assists": 3,
            "passes_attempted": 800,
            "passes_completed": 720,
            "tackles": 50,
            "interceptions": 30,
            "sprints": 200,
            "distance_covered_km": 300.0,
            "total_injuries": 1
        }
        
        response = client.post("/predict", json=player_data)
        
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data or "risk_level" in data
            assert "confidence" in data or "error" in data
    
    def test_predict_with_missing_fields(self):
        """Test prediction with missing required fields."""
        incomplete_data = {
            "age": 25,
            "matches_played": 30
        }
        
        response = client.post("/predict", json=incomplete_data)
        
        # Should still work as fields have defaults
        assert response.status_code in [200, 422, 503]


@pytest.mark.skipif(not API_AVAILABLE, reason="FastAPI not available")
class TestUploadEndpoint:
    """Test video upload endpoint."""
    
    def test_upload_invalid_file(self):
        """Test upload with invalid file type."""
        # Create dummy text file
        files = {"file": ("test.txt", "dummy content", "text/plain")}
        response = client.post("/upload", files=files)
        
        assert response.status_code == 400
    
    def test_upload_without_file(self):
        """Test upload without file."""
        response = client.post("/upload")
        
        assert response.status_code == 422


@pytest.mark.skipif(not API_AVAILABLE, reason="FastAPI not available")
class TestMetricsEndpoint:
    """Test metrics endpoint."""
    
    def test_get_metrics(self):
        """Test /metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)


@pytest.mark.skipif(not API_AVAILABLE, reason="FastAPI not available")
class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


def test_root_available():
    """Basic test to ensure tests can run."""
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

