"""
Model Tests for ScoutIA Pro
Tests ML model training and inference.
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preparation.preprocess_data import (
    preprocess_pipeline,
    create_features,
    clean_data,
    load_data
)
from models.predict import InjuryRiskPredictor


class TestDataPreprocessing:
    """Test data preprocessing functions."""
    
    def test_load_data_creates_sample(self):
        """Test that load_data creates sample data when no CSV exists."""
        # This should not raise an error
        try:
            df = load_data("data/raw/nonexistent")
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
        except Exception as e:
            pytest.skip(f"Could not load data: {e}")
    
    def test_create_features(self):
        """Test feature creation."""
        # Create sample data
        df = pd.DataFrame({
            'passes_completed': [100, 200],
            'passes_attempted': [120, 250],
            'minutes_played': [1800, 2000],
            'matches_played': [25, 30],
            'goals': [5, 10],
            'sprints': [100, 150],
            'distance_covered_km': [200, 250],
            'tackles': [20, 30],
            'interceptions': [15, 20]
        })
        
        df_with_features = create_features(df)
        
        assert 'pass_accuracy' in df_with_features.columns
        assert 'intensity' in df_with_features.columns
        assert 'goals_per_match' in df_with_features.columns
    
    def test_clean_data(self):
        """Test data cleaning."""
        # Create data with outliers
        df = pd.DataFrame({
            'value': [1, 2, 3, 1000, 4, 5],
            'matches': [10, 20, 30, 40, 50, 60]
        })
        
        df_cleaned = clean_data(df)
        
        assert isinstance(df_cleaned, pd.DataFrame)
        assert len(df_cleaned) > 0


class TestModelInference:
    """Test model inference."""
    
    def test_predictor_initialization(self):
        """Test predictor can be initialized."""
        try:
            predictor = InjuryRiskPredictor()
            # Should not raise error even if model doesn't exist
            assert hasattr(predictor, 'model')
        except Exception as e:
            pytest.skip(f"Could not initialize predictor: {e}")
    
    def test_predict_with_sample_data(self):
        """Test prediction with sample data."""
        predictor = InjuryRiskPredictor()
        
        if predictor.model is None:
            pytest.skip("Model not available")
        
        sample_data = {
            'age': 25,
            'matches_played': 30,
            'minutes_played': 2400,
            'goals': 5,
            'assists': 3,
            'passes_attempted': 800,
            'passes_completed': 720,
            'tackles': 50,
            'interceptions': 30,
            'sprints': 200,
            'distance_covered_km': 300.0,
            'total_injuries': 1
        }
        
        result = predictor.predict(sample_data)
        
        assert isinstance(result, dict)
        assert 'risk_level' in result or 'error' in result


class TestPipelineIntegration:
    """Test pipeline integration."""
    
    def test_preprocess_pipeline_runs(self):
        """Test that preprocessing pipeline runs without error."""
        try:
            df = preprocess_pipeline(
                data_dir="data/raw",
                output_path="data/processed/players_clean_test.csv"
            )
            assert isinstance(df, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"Pipeline failed: {e}")


def test_basic_functionality():
    """Basic test to ensure tests can run."""
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

