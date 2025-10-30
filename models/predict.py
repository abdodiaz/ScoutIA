"""
ML Model Prediction Module for ScoutIA Pro
Provides inference capabilities for injury risk prediction.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InjuryRiskPredictor:
    """Predictor class for injury risk assessment."""
    
    def __init__(self, model_path: str = "models/injury_risk_model.pkl"):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to saved model file
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        
        if self.model_path.exists():
            self.load_model()
        else:
            logger.error(f"Model not found at {model_path}")
    
    def load_model(self):
        """Load the trained model and feature names."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            
            # Load feature names
            feature_names_path = self.model_path.parent / "feature_names.txt"
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(self.feature_names)} feature names")
            else:
                logger.warning("Feature names file not found")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def predict(self, data: dict or pd.DataFrame) -> dict:
        """
        Predict injury risk from player data.
        
        Args:
            data: Dictionary or DataFrame with player features
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {
                "error": "Model not loaded",
                "prediction": None,
                "confidence": None,
                "risk_level": None
            }
        
        # Convert dict to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns) if self.feature_names else set()
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Fill missing features with default values
            for feat in missing_features:
                df[feat] = 0
        
        # Select and order features
        if self.feature_names:
            df = df[self.feature_names]
        
        # Predict
        try:
            prediction = self.model.predict(df)[0]
            probabilities = self.model.predict_proba(df)[0]
            
            # Map prediction to risk level
            risk_levels = ['Low', 'Medium', 'High']
            risk_level = risk_levels[int(prediction)] if prediction < len(risk_levels) else 'Unknown'
            confidence = float(np.max(probabilities))
            
            return {
                "prediction": int(prediction),
                "risk_level": risk_level,
                "confidence": confidence,
                "probabilities": {
                    "low": float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                    "medium": float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                    "high": float(probabilities[2]) if len(probabilities) > 2 else 0.0,
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "prediction": None,
                "confidence": None,
                "risk_level": None
            }
    
    def predict_from_json(self, json_str: str) -> dict:
        """
        Predict from JSON string input.
        
        Args:
            json_str: JSON string with player data
            
        Returns:
            Dictionary with prediction results
        """
        try:
            data = json.loads(json_str)
            return self.predict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return {
                "error": f"Invalid JSON: {e}",
                "prediction": None,
                "confidence": None,
                "risk_level": None
            }
    
    def batch_predict(self, data_list: list) -> list:
        """
        Predict injury risk for multiple players.
        
        Args:
            data_list: List of dictionaries or DataFrame with multiple players
            
        Returns:
            List of prediction results
        """
        results = []
        for data in data_list:
            result = self.predict(data)
            results.append(result)
        return results


def main():
    """Example usage of the predictor."""
    # Initialize predictor
    predictor = InjuryRiskPredictor()
    
    if predictor.model is None:
        logger.error("Cannot proceed without a trained model. Please run train_model.py first.")
        return
    
    # Example prediction
    example_data = {
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
        'distance_covered_km': 300,
        'total_injuries': 1,
        'pass_accuracy': 90.0,
        'intensity': 80.0,
        'goals_per_match': 0.17,
        'distance_per_match': 10.0,
        'sprint_per_match': 6.67,
        'defensive_activity': 80
    }
    
    result = predictor.predict(example_data)
    
    logger.info("\n" + "="*50)
    logger.info("Prediction Result:")
    logger.info(f"Risk Level: {result['risk_level']}")
    logger.info(f"Confidence: {result['confidence']:.2%}")
    logger.info(f"Probabilities:")
    logger.info(f"  - Low: {result['probabilities']['low']:.2%}")
    logger.info(f"  - Medium: {result['probabilities']['medium']:.2%}")
    logger.info(f"  - High: {result['probabilities']['high']:.2%}")
    logger.info("="*50)


if __name__ == "__main__":
    main()

