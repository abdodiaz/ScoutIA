"""
ML Model Training Module for ScoutIA Pro
Trains RandomForest model for injury risk prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path
import logging
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preparation.preprocess_data import preprocess_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_path: str = "data/processed/players_clean.csv") -> pd.DataFrame:
    """
    Load preprocessed training data.
    
    Args:
        data_path: Path to processed data CSV
        
    Returns:
        DataFrame with training data
    """
    data_file = Path(data_path)
    
    if not data_file.exists():
        logger.warning(f"Processed data not found at {data_path}. Running preprocessing...")
        df = preprocess_pipeline()
    else:
        df = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(df)} training samples")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for model training.
    
    Args:
        df: Training DataFrame
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    # Define feature columns (excluding non-feature columns)
    exclude_cols = ['player_id', 'injury_risk', 'position']
    
    # Check which columns exist
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['injury_risk_numeric'] if 'injury_risk_numeric' in df.columns else None
    
    # If no target column exists, create one based on total_injuries
    if y is None:
        if 'total_injuries' in df.columns:
            y = pd.cut(df['total_injuries'], bins=[-1, 0, 2, 100], labels=[0, 1, 2])
            y = y.astype(int)
        else:
            logger.error("No target variable found in data")
            return None, None
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, 
                model_path: str = "models/injury_risk_model.pkl",
                test_size: float = 0.2,
                random_state: int = 42) -> dict:
    """
    Train RandomForest model for injury risk prediction.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_path: Path to save trained model
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Dictionary with model metrics
    """
    logger.info("Starting model training...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, 
                                      target_names=['Low', 'Medium', 'High']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Most Important Features:")
    logger.info(feature_importance.head(10).to_string(index=False))
    
    # Save model
    model_path_obj = Path(model_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save feature names for inference
    feature_names_path = model_path_obj.parent / "feature_names.txt"
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(X.columns))
    logger.info(f"Feature names saved to {feature_names_path}")
    
    return {
        'accuracy': accuracy,
        'model': model,
        'feature_importance': feature_importance,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, 
                                                       target_names=['Low', 'Medium', 'High'])
    }


def main():
    """Main training pipeline."""
    logger.info("="*50)
    logger.info("ScoutIA Pro - Model Training")
    logger.info("="*50)
    
    # Load data
    df = load_training_data()
    
    # Prepare features
    X, y = prepare_features(df)
    
    if X is None or y is None:
        logger.error("Failed to prepare features and target")
        return
    
    # Train model
    results = train_model(X, y)
    
    logger.info("="*50)
    logger.info("Training completed successfully!")
    logger.info("="*50)


if __name__ == "__main__":
    main()

