"""
Data preprocessing module for ScoutIA Pro
Loads, cleans, and normalizes football player data from CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load football player data from CSV files in the raw data directory.
    
    Args:
        data_dir: Path to raw data directory
        
    Returns:
        DataFrame with player data
    """
    data_path = Path(data_dir)
    
    # Look for CSV files in the directory
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}. Creating sample data...")
        return create_sample_data()
    
    # Load first CSV found (can be extended to load multiple)
    df = pd.read_csv(csv_files[0])
    logger.info(f"Loaded {len(df)} rows from {csv_files[0]}")
    
    return df


def create_sample_data() -> pd.DataFrame:
    """
    Create sample football player data for testing purposes.
    
    Returns:
        DataFrame with sample player statistics
    """
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'player_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 36, n_samples),
        'position': np.random.choice(['GK', 'DF', 'MF', 'FW'], n_samples),
        'matches_played': np.random.randint(10, 40, n_samples),
        'minutes_played': np.random.randint(500, 3000, n_samples),
        'goals': np.random.randint(0, 20, n_samples),
        'assists': np.random.randint(0, 15, n_samples),
        'passes_attempted': np.random.randint(200, 1500, n_samples),
        'passes_completed': np.random.randint(150, 1200, n_samples),
        'tackles': np.random.randint(20, 150, n_samples),
        'interceptions': np.random.randint(10, 80, n_samples),
        'sprints': np.random.randint(50, 400, n_samples),
        'distance_covered_km': np.random.uniform(80, 400, n_samples).round(2),
        'total_injuries': np.random.randint(0, 5, n_samples),
        'injury_risk': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.6, 0.3, 0.1])
    }
    
    return pd.DataFrame(data)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw player data.
    
    Args:
        df: Raw player data DataFrame
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Pass accuracy
    if 'passes_completed' in df.columns and 'passes_attempted' in df.columns:
        df['pass_accuracy'] = (df['passes_completed'] / df['passes_attempted'] * 100).fillna(0)
    else:
        df['pass_accuracy'] = np.random.uniform(65, 95, len(df))
    
    # Intensity score (minutes per match)
    if 'minutes_played' in df.columns and 'matches_played' in df.columns:
        df['intensity'] = (df['minutes_played'] / df['matches_played']).fillna(0)
    else:
        df['intensity'] = np.random.uniform(60, 90, len(df))
    
    # Goals per match
    if 'goals' in df.columns and 'matches_played' in df.columns:
        df['goals_per_match'] = (df['goals'] / df['matches_played']).fillna(0)
    else:
        df['goals_per_match'] = np.random.uniform(0, 1.2, len(df))
    
    # Distance per match
    if 'distance_covered_km' in df.columns and 'matches_played' in df.columns:
        df['distance_per_match'] = (df['distance_covered_km'] / df['matches_played']).fillna(0)
    else:
        df['distance_per_match'] = np.random.uniform(8, 12, len(df))
    
    # Sprint intensity
    if 'sprints' in df.columns and 'matches_played' in df.columns:
        df['sprint_per_match'] = (df['sprints'] / df['matches_played']).fillna(0)
    else:
        df['sprint_per_match'] = np.random.uniform(5, 25, len(df))
    
    # Defensive activity
    if 'tackles' in df.columns and 'interceptions' in df.columns:
        df['defensive_activity'] = df['tackles'] + df['interceptions']
    else:
        df['defensive_activity'] = np.random.randint(30, 200, len(df))
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize player data.
    
    Args:
        df: Raw data DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        logger.info(f"Removed {initial_len - len(df)} duplicate rows")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Handle outliers in numeric columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Normalize injury_risk to numeric (if exists)
    if 'injury_risk' in df.columns:
        risk_mapping = {'low': 0, 'medium': 1, 'high': 2}
        df['injury_risk_numeric'] = df['injury_risk'].map(risk_mapping).fillna(0)
    
    logger.info(f"Data cleaning complete. Final dataset: {len(df)} rows")
    
    return df


def preprocess_pipeline(data_dir: str = "data/raw", 
                       output_path: str = "data/processed/players_clean.csv") -> pd.DataFrame:
    """
    Complete preprocessing pipeline: load → clean → create features → save.
    
    Args:
        data_dir: Input directory for raw data
        output_path: Output path for processed data
        
    Returns:
        Processed DataFrame
    """
    logger.info("Starting data preprocessing pipeline...")
    
    # Load data
    df = load_data(data_dir)
    
    # Clean data
    df = clean_data(df)
    
    # Create features
    df = create_features(df)
    
    # Save processed data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Processed data saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Run the preprocessing pipeline
    processed_df = preprocess_pipeline()
    print("\nProcessed Data Summary:")
    print(processed_df.describe())
    print("\nFirst 5 rows:")
    print(processed_df.head())

