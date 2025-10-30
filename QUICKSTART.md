# 🚀 ScoutIA Pro - Quick Start Guide

## 5-Minute Setup

### 1️⃣ Install Dependencies

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2️⃣ Run Pipeline

```bash
# Complete setup (data prep + training)
python scripts/run_pipeline.py
```

### 3️⃣ Launch Dashboard

```bash
# Streamlit Dashboard
streamlit run frontend/streamlit_app.py
```

Access at: **http://localhost:8501**

### 4️⃣ Test Prediction

In the Streamlit dashboard:
1. Go to **"📊 Player Analysis"** tab
2. Enter player data
3. Click **"🔮 Predict Injury Risk"**

## 🎯 Try It Now

### Quick Test via Python

```python
from models.predict import InjuryRiskPredictor
from models.extract_features_from_video import VideoFeatureExtractor

# Initialize
predictor = InjuryRiskPredictor()

# Predict
data = {
    'age': 25, 'matches_played': 30, 'minutes_played': 2400,
    'goals': 5, 'assists': 3, 'passes_attempted': 800,
    'passes_completed': 720, 'tackles': 50, 'interceptions': 30,
    'sprints': 200, 'distance_covered_km': 300.0, 'total_injuries': 1
}

result = predictor.predict(data)
print(f"Risk: {result['risk_level']}, Confidence: {result['confidence']:.2%}")
```

## 📹 Analyze Video

```python
# Extract features from video
extractor = VideoFeatureExtractor()
df = extractor.extract_features("data/videos/raw/08fd33_4.mp4")

print(df[['speed', 'acceleration', 'num_players']].describe())
```

## 🔌 Use API

```bash
# Start API
uvicorn backend.main:app --reload

# Test in browser
# http://localhost:8000/docs
```

## ✅ Checklist

- [ ] Dependencies installed
- [ ] Pipeline run successfully
- [ ] Model trained and saved
- [ ] Dashboard running
- [ ] First prediction made

## 🆘 Need Help?

See full documentation: `README_SETUP.md`

