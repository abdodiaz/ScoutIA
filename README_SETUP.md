# ScoutIA Pro - Setup Guide

## 📋 Overview

ScoutIA Pro is an AI-powered football player performance analysis and injury risk prediction system. It combines Machine Learning, Computer Vision, FastAPI, and Streamlit to provide comprehensive analytics.

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (optional, for faster inference)
- 16GB RAM recommended
- Windows 10/11 or Linux

### Step 1: Clone and Setup Environment

```bash
# Navigate to project directory
cd ScoutIA

# Create virtual environment (if not already created)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download YOLO Model

The YOLOv8 model will be automatically downloaded on first use, or you can manually place it in the `YOLO/` directory.

```bash
# YOLO model will be auto-downloaded, or download from:
# https://github.com/ultralytics/ultralytics
```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline

```bash
# Run entire pipeline (data prep → training → testing)
python scripts/run_pipeline.py

# With API launch
python scripts/run_pipeline.py --launch-api
```

### Option 2: Run Individual Components

#### Data Preprocessing
```bash
python src/data_preparation/preprocess_data.py
```

#### Model Training
```bash
python models/train_model.py
```

#### Model Prediction
```bash
python models/predict.py
```

#### Video Feature Extraction
```bash
python models/extract_features_from_video.py path/to/video.mp4
```

### Option 3: Launch Services

#### FastAPI Backend
```bash
cd backend
uvicorn main:app --reload

# API will be available at:
# - http://localhost:8000
# - Docs: http://localhost:8000/docs
```

#### Streamlit Dashboard
```bash
streamlit run frontend/streamlit_app.py

# Dashboard will be available at:
# http://localhost:8501
```

## 📁 Project Structure

```
ScoutIA/
├── backend/
│   ├── main.py              # FastAPI backend
│   └── requirements.txt
├── models/
│   ├── train_model.py       # ML model training
│   ├── predict.py           # ML inference
│   ├── yolo_infer.py        # YOLO detection
│   ├── pose_estimation.py   # MediaPipe pose
│   └── extract_features_from_video.py  # Video analysis
├── src/
│   └── data_preparation/
│       └── preprocess_data.py  # Data preprocessing
├── frontend/
│   ├── streamlit_app.py     # Dashboard UI
│   └── index.html           # Placeholder
├── scripts/
│   └── run_pipeline.py      # Complete pipeline
├── tests/
│   ├── test_api.py          # API tests
│   ├── test_model.py        # Model tests
│   └── test_sample.py
├── data/
│   ├── raw/                 # Raw data
│   ├── processed/           # Processed data
│   └── videos/              # Video files
└── requirements.txt
```

## 🎯 Usage Examples

### 1. Player Injury Risk Prediction

```python
from models.predict import InjuryRiskPredictor

predictor = InjuryRiskPredictor()

player_data = {
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

result = predictor.predict(player_data)
print(f"Risk Level: {result['risk_level']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 2. Video Analysis

```python
from models.extract_features_from_video import VideoFeatureExtractor

extractor = VideoFeatureExtractor()
df = extractor.extract_features("path/to/video.mp4", "output.csv")
print(df.describe())
```

### 3. API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict injury risk
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 25, "matches_played": 30, ...}'

# Upload video
curl -X POST http://localhost:8000/upload \
  -F "file=@video.mp4"
```

## 🧪 Testing

Run all tests:
```bash
pytest tests/
```

Run specific test files:
```bash
pytest tests/test_api.py -v
pytest tests/test_model.py -v
```

## 📊 Features

### Data Processing
- ✅ CSV data loading and cleaning
- ✅ Feature engineering (intensity, pass_accuracy, etc.)
- ✅ Automatic normalization
- ✅ Outlier detection and handling

### Machine Learning
- ✅ Random Forest Classifier
- ✅ Injury risk prediction (Low/Medium/High)
- ✅ Feature importance analysis
- ✅ Model persistence (PKL format)

### Computer Vision
- ✅ YOLOv8 object detection (players, ball)
- ✅ MediaPipe pose estimation
- ✅ Motion metrics (speed, acceleration)
- ✅ Biomechanical analysis (joint angles)

### API & Dashboard
- ✅ FastAPI REST API
- ✅ Streamlit interactive dashboard
- ✅ Real-time predictions
- ✅ Video upload and analysis
- ✅ CSV batch processing

## 🔧 Configuration

### GPU Setup (Optional)

For NVIDIA GPU acceleration:

```python
# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Environment Variables

Create a `.env` file:
```env
ENV=dev
MODEL_PATH=models/injury_risk_model.pkl
DATA_PATH=data/processed
```

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Predict injury risk |
| `/upload` | POST | Upload and analyze video |
| `/metrics` | GET | Model metrics |
| `/docs` | GET | API documentation |

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**
   - Solution: Run `python models/train_model.py` to train the model

2. **Import errors**
   - Solution: Ensure virtual environment is activated and dependencies are installed

3. **GPU not detected**
   - Solution: Install PyTorch with CUDA support or use CPU mode

4. **Memory errors**
   - Solution: Reduce batch size or use smaller videos

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit pull request

## 📄 License

This project is for educational purposes.

## 📧 Support

For issues or questions, please open an issue on GitHub.

## 🎉 Acknowledgments

- Ultralytics YOLO for object detection
- MediaPipe for pose estimation
- FastAPI and Streamlit teams
- Scikit-learn for ML algorithms

