
# 🧠 **PROMPT POUR CURSOR – Projet ScoutIA Pro (Scouting Football Master Pro)**

## 🎯 **CONTEXTE DU PROJET**
Je développe une application IA locale appelée **ScoutIA Pro**, inspirée des solutions professionnelles (Hudl, Wyscout, FIFA Performance).  
Elle sert à **analyser la performance des joueurs de football** et à **prédire les risques de blessure**, à partir de **données vidéo et statistiques**.

Le projet fonctionne **100% en local**, sur ma machine :
> 💻 Intel i7-10510U, 16 Go RAM, GPU NVIDIA MX130 (2 Go VRAM)  
> 🧩 Aucun cloud — uniquement environnement local Docker/FastAPI/Streamlit  

## ⚙️ **STRUCTURE EXISTANTE DU PROJET**
```
ScoutIA-Pro/
├── backend/
│   ├── main.py
│   ├── requirements.txt
├── models/
│   ├── yolo_infer.py
│   ├── pose_estimation.py
│   ├── train_model.py
│   ├── predict.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── annotations/
├── scripts/
│   ├── run_pipeline.py
├── frontend/
│   ├── index.html
│   ├── streamlit_app.py
├── tests/
│   ├── test_sample.py
└── docker-compose.yml
```

## 🧩 **OBJECTIF À ATTEINDRE**
Compléter **tous les fichiers manquants** pour rendre le projet *ScoutIA Pro* totalement fonctionnel :
- Préparation des données  
- Entraînement ML  
- Vision par ordinateur  
- API FastAPI  
- Dashboard Streamlit  
- Pipeline d’automatisation  

## 🔹 **1️⃣ DATA ENGINEERING**
Créer `src/data_preparation/preprocess_data.py` :
- Charger les CSV depuis `/data/raw`
- Nettoyer et normaliser
- Créer les features (`intensity`, `pass_accuracy`)
- Sauvegarder `/data/processed/players_clean.csv`

## 🔹 **2️⃣ MACHINE LEARNING**
Créer `models/train_model.py` et `models/predict.py` :
- Entraîner modèle ML (RandomForestClassifier ou LogisticRegression)
- Sauvegarder en `.pkl`
- Script de prédiction basé sur JSON input

## 🔹 **3️⃣ COMPUTER VISION**
Créer `models/extract_features_from_video.py` :
- YOLOv8 pour détection joueurs + ballon
- MediaPipe pour angles articulaires
- Extraire (x, y, speed, acceleration, angles)
- Sauvegarder `/data/processed/video_features.csv`

## 🔹 **4️⃣ BACKEND API (FASTAPI)**
Compléter `backend/main.py` :
```
/health → statut API
/predict → prédiction modèle ML
/upload → upload vidéo + traitement YOLO
/metrics → scores du modèle
```

## 🔹 **5️⃣ FRONTEND (STREAMLIT DASHBOARD)**
Créer `frontend/streamlit_app.py` :
- Upload vidéo ou CSV
- Visualisation : heatmap, stats, risque blessure
- Graphiques interactifs (Plotly, Seaborn)

## 🔹 **6️⃣ PIPELINE AUTOMATISÉ**
Compléter `scripts/run_pipeline.py` :
- Charger → Prétraiter → Entraîner → Tester → Rapport PDF → Lancer API

## 🔹 **7️⃣ TESTS & QUALITÉ**
Ajouter `tests/test_api.py`, `tests/test_model.py` :
- Tester endpoints API
- Vérifier précision modèle
- Tester pipeline complet

## ⚙️ **CONTRAINTES TECHNIQUES**
- Fonctionnement 100% local
- Frameworks : `pandas`, `numpy`, `scikit-learn`, `mediapipe`, `ultralytics`, `fastapi`, `streamlit`, `plotly`, `matplotlib`
- Optimisation pour GPU MX130

## 🚀 **LIVRABLE FINAL**
- ✅ API FastAPI
- ✅ Dashboard Streamlit
- ✅ Modèle ML entraîné localement
- ✅ YOLO/MediaPipe opérationnels
- ✅ Pipeline complet
- ✅ Documentation claire

## 💬 **INSTRUCTION À CURSOR**
En te basant sur cette structure et ces instructions :
- Crée tous les fichiers manquants
- Complète les scripts existants
- Documente et commente le code
- Exemples d’entrée/sortie
- Code lisible, modulaire, PEP8

## 🧾 **Résumé en une phrase**
> Développe l’application complète *ScoutIA Pro* — une solution IA locale pour l’analyse et la prédiction de performance footballistique, combinant Machine Learning, Computer Vision, FastAPI et Streamlit, selon la structure existante et les contraintes matérielles.
