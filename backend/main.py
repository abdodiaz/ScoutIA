"""
FastAPI Backend for ScoutIA Pro
Provides REST API endpoints for injury risk prediction and video analysis.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import sys
from pathlib import Path
import logging
import uvicorn

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.predict import InjuryRiskPredictor
from models.extract_features_from_video import VideoFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ScoutIA Pro API",
    description="AI-powered football player performance analysis and injury risk prediction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (lazy loading)
predictor = None
video_extractor = None


def get_predictor():
    """Lazy load predictor."""
    global predictor
    if predictor is None:
        predictor = InjuryRiskPredictor()
    return predictor


def get_video_extractor():
    """Lazy load video extractor."""
    global video_extractor
    if video_extractor is None:
        video_extractor = VideoFeatureExtractor()
    return video_extractor


# Request models
class PlayerData(BaseModel):
    """Player data for injury risk prediction."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )
    
    age: int = 25
    matches_played: int = 30
    minutes_played: int = 2400
    goals: int = 5
    assists: int = 3
    passes_attempted: int = 800
    passes_completed: int = 720
    tackles: int = 50
    interceptions: int = 30
    sprints: int = 200
    distance_covered_km: float = 300.0
    total_injuries: int = 1


# API Endpoints
@app.get("/health")
def health():
    """Health check endpoint."""
    return JSONResponse({
        "status": "ok",
        "service": "ScoutIA Pro API",
        "version": "1.0.0"
    })


@app.post("/predict")
async def predict_injury_risk(player_data: PlayerData):
    """
    Predict injury risk for a player based on performance data.
    
    Args:
        player_data: Player performance statistics
        
    Returns:
        Prediction result with risk level and confidence
    """
    try:
        predictor = get_predictor()
        
        if predictor.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train a model first."
            )
        
        # Convert to dict
        data = player_data.model_dump()
        
        # Add derived features if not present
        if 'pass_accuracy' not in data:
            data['pass_accuracy'] = (data['passes_completed'] / data['passes_attempted'] * 100) if data['passes_attempted'] > 0 else 0
        
        if 'intensity' not in data:
            data['intensity'] = (data['minutes_played'] / data['matches_played']) if data['matches_played'] > 0 else 0
        
        if 'goals_per_match' not in data:
            data['goals_per_match'] = (data['goals'] / data['matches_played']) if data['matches_played'] > 0 else 0
        
        if 'distance_per_match' not in data:
            data['distance_per_match'] = (data['distance_covered_km'] / data['matches_played']) if data['matches_played'] > 0 else 0
        
        if 'sprint_per_match' not in data:
            data['sprint_per_match'] = (data['sprints'] / data['matches_played']) if data['matches_played'] > 0 else 0
        
        if 'defensive_activity' not in data:
            data['defensive_activity'] = data.get('tackles', 0) + data.get('interceptions', 0)
        
        # Get prediction
        result = predictor.predict(data)
        
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_and_analyze_video(file: UploadFile = File(...)):
    """
    Upload a video file and extract features using YOLO and MediaPipe.
    
    Args:
        file: Video file upload
        
    Returns:
        Analysis result with extracted features
    """
    try:
        # Check file type
        if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a video file."
            )
        
        # Save uploaded file
        upload_dir = Path("data/videos/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = upload_dir / file.filename
        
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Saved uploaded video to {video_path}")
        
        # Extract features
        extractor = get_video_extractor()
        output_path = f"data/processed/{file.filename}_features.csv"
        
        df = extractor.extract_features(str(video_path), output_path)
        
        # Calculate summary statistics
        summary = {
            "num_frames": len(df),
            "avg_players": float(df['num_players'].mean()) if 'num_players' in df.columns else 0,
            "ball_detection_rate": float(df['ball_detected'].mean()) if 'ball_detected' in df.columns else 0,
            "avg_speed": float(df['speed'].mean()) if 'speed' in df.columns else 0,
            "max_speed": float(df['speed'].max()) if 'speed' in df.columns else 0,
            "avg_acceleration": float(df['acceleration'].mean()) if 'acceleration' in df.columns else 0,
            "features_csv": output_path
        }
        
        return JSONResponse({
            "status": "success",
            "video": file.filename,
            "summary": summary,
            "features_path": output_path
        })
    
    except HTTPException:
        # Re-raise HTTP exceptions (like 400 for invalid file type)
        raise
    except Exception as e:
        logger.error(f"Video upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_model_metrics():
    """
    Get model performance metrics.
    
    Returns:
        Model metrics and feature importance
    """
    try:
        predictor = get_predictor()
        
        if predictor.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train a model first."
            )
        
        # Load metrics from training (if available)
        metrics_file = Path("models/metrics.json")
        
        if metrics_file.exists():
            import json
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {
                "accuracy": None,
                "message": "No metrics file found. Train a model to generate metrics."
            }
        
        return JSONResponse(metrics)
    
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "ScoutIA Pro API",
        "version": "1.0.0",
        "description": "AI-powered football player analysis and injury risk prediction",
        "endpoints": {
            "/health": "Health check",
            "/predict": "POST - Predict injury risk",
            "/upload": "POST - Upload and analyze video",
            "/metrics": "GET - Model performance metrics",
            "/docs": "API documentation"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

