# 🚀 ScoutIA Pro - How to Run

## ✅ **Quick Start (All Services)**

### Prerequisites
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### Option 1: Run Everything at Once

#### Terminal 1: Streamlit Dashboard
```powershell
streamlit run frontend/streamlit_app.py
```
**Open**: http://localhost:8501

#### Terminal 2: FastAPI Backend
```powershell
cd backend
uvicorn main:app --reload
```
**Open**: http://localhost:8000/docs

---

## 📋 **Individual Services**

### 1. Run Complete Pipeline
```powershell
python scripts/run_pipeline.py
```
This will:
- ✅ Preprocess data
- ✅ Train model
- ✅ Evaluate model
- ✅ Generate report

### 2. Train Model Only
```powershell
python models/train_model.py
```

### 3. Run Tests
```powershell
pytest tests/ -v
```

### 4. Test API Endpoints
```powershell
# Health check
Invoke-RestMethod -Uri http://localhost:8000/health

# Prediction test
Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST -ContentType "application/json" -Body '{"age":25,"matches_played":30,"minutes_played":2400,"goals":5,"assists":3,"passes_attempted":800,"passes_completed":720,"tackles":50,"interceptions":30,"sprints":200,"distance_covered_km":300,"total_injuries":1}'
```

---

## 🌐 **Access Services**

### Streamlit Dashboard
- **URL**: http://localhost:8501
- **Features**: Player analysis, Video analysis, Model info

### FastAPI Backend
- **URL**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

---

## 🎯 **Usage Examples**

### Python Script Example
```python
from models.predict import InjuryRiskPredictor

# Initialize predictor
predictor = InjuryRiskPredictor()

# Make prediction
result = predictor.predict({
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
})

print(f"Risk Level: {result['risk_level']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Video Analysis Example
```python
from models.extract_features_from_video import VideoFeatureExtractor

extractor = VideoFeatureExtractor()
df = extractor.extract_features("data/videos/raw/08fd33_4.mp4")
print(df.describe())
```

---

## 🔧 **Troubleshooting**

### Model Not Found
```powershell
# Train the model first
python models/train_model.py
```

### Import Errors
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Port Already in Use
```powershell
# Kill existing processes
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process

# Or change ports in:
# - Streamlit: streamlit run app.py --server.port 8502
# - FastAPI: uvicorn main:app --port 8001
```

---

## 📊 **Test Results**

All 16 tests passing:
- ✅ 8 API tests
- ✅ 6 Model tests  
- ✅ 2 Pipeline tests

Run tests:
```powershell
pytest tests/ -v
```

---

## 🎉 **Success Criteria**

✅ All tests pass  
✅ Services start without errors  
✅ API responds to health check  
✅ Dashboard loads in browser  
✅ Model can make predictions  
✅ Pipeline completes successfully  

**Status**: ✅ All criteria met!

---

**Happy analyzing! ⚽🚀**

