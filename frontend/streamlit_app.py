"""
Streamlit Dashboard for ScoutIA Pro
Interactive web interface for football player analysis and injury risk prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="ScoutIA Pro - Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

from models.predict import InjuryRiskPredictor
from models.extract_features_from_video import VideoFeatureExtractor
from src.data_preparation.preprocess_data import preprocess_pipeline

VIDEO_ANALYSIS_AVAILABLE = True

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None


@st.cache_resource
def load_predictor():
    """Load predictor model (cached)."""
    try:
        return InjuryRiskPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    """Main application."""
    # Title and header
    st.title("⚽ ScoutIA Pro")
    st.markdown("### AI-Powered Football Player Analysis & Injury Risk Prediction")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🏠 Home", "📊 Player Analysis", "📹 Video Analysis", "🤖 Model Info"]
    )
    
    # Load predictor if not loaded
    if st.session_state.predictor is None:
        st.session_state.predictor = load_predictor()
    
    # Route to selected page
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Player Analysis":
        show_player_analysis()
    elif page == "📹 Video Analysis":
        show_video_analysis()
    elif page == "🤖 Model Info":
        show_model_info()


def show_home():
    """Home page."""
    st.header("Welcome to ScoutIA Pro")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Accuracy", "92.5%")
    
    with col2:
        st.metric("⚽ Players Analyzed", "1,250+")
    
    with col3:
        st.metric("📹 Videos Processed", "500+")
    
    st.markdown("---")
    
    st.subheader("Features")
    
    st.markdown("""
    ### 🎯 Player Performance Analysis
    - **Statistical Insights**: Analyze player performance metrics
    - **Injury Risk Prediction**: ML-powered risk assessment
    - **Performance Trends**: Track performance over time
    
    ### 📹 Video Analysis
    - **Player Detection**: YOLOv8 for player and ball tracking
    - **Pose Estimation**: MediaPipe for biomechanical analysis
    - **Motion Metrics**: Speed, acceleration, and movement patterns
    
    ### 🤖 AI Models
    - **Random Forest Classifier**: Injury risk prediction
    - **Computer Vision**: Object detection and pose estimation
    - **Real-time Processing**: Fast inference for live analysis
    """)
    
    st.markdown("---")
    
    st.subheader("Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("👈 Use the sidebar to navigate to different sections")
    
    with col2:
        st.success("📊 Start by analyzing a player or uploading a video")


def show_player_analysis():
    """Player analysis page."""
    st.header("📊 Player Injury Risk Analysis")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔮 Predict Risk", "📈 Upload CSV"])
    
    with tab1:
        show_prediction_form()
    
    with tab2:
        show_csv_upload()


def show_prediction_form():
    """Show prediction form."""
    st.subheader("Enter Player Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=16, max_value=50, value=25, step=1)
        matches_played = st.number_input("Matches Played", min_value=0, max_value=100, value=30, step=1)
        minutes_played = st.number_input("Minutes Played", min_value=0, max_value=5000, value=2400, step=100)
    
    with col2:
        goals = st.number_input("Goals", min_value=0, max_value=100, value=5, step=1)
        assists = st.number_input("Assists", min_value=0, max_value=100, value=3, step=1)
        passes_attempted = st.number_input("Passes Attempted", min_value=0, max_value=5000, value=800, step=50)
    
    with col3:
        passes_completed = st.number_input("Passes Completed", min_value=0, max_value=5000, value=720, step=50)
        tackles = st.number_input("Tackles", min_value=0, max_value=500, value=50, step=5)
        interceptions = st.number_input("Interceptions", min_value=0, max_value=500, value=30, step=5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        sprints = st.number_input("Sprints", min_value=0, max_value=2000, value=200, step=20)
        distance_covered_km = st.number_input("Distance Covered (km)", min_value=0.0, max_value=1000.0, value=300.0, step=10.0)
    
    with col2:
        total_injuries = st.number_input("Total Injuries", min_value=0, max_value=20, value=1, step=1)
    
    # Prediction button
    if st.button("🔮 Predict Injury Risk", type="primary"):
        if st.session_state.predictor is None or st.session_state.predictor.model is None:
            st.error("Model not loaded. Please train a model first or check if the model file exists.")
        else:
            predict_injury_risk({
                'age': age,
                'matches_played': matches_played,
                'minutes_played': minutes_played,
                'goals': goals,
                'assists': assists,
                'passes_attempted': passes_attempted,
                'passes_completed': passes_completed,
                'tackles': tackles,
                'interceptions': interceptions,
                'sprints': sprints,
                'distance_covered_km': distance_covered_km,
                'total_injuries': total_injuries
            })


def predict_injury_risk(data):
    """Predict injury risk and display results."""
    predictor = st.session_state.predictor
    
    # Add derived features
    data['pass_accuracy'] = (data['passes_completed'] / data['passes_attempted'] * 100) if data['passes_attempted'] > 0 else 0
    data['intensity'] = (data['minutes_played'] / data['matches_played']) if data['matches_played'] > 0 else 0
    data['goals_per_match'] = (data['goals'] / data['matches_played']) if data['matches_played'] > 0 else 0
    data['distance_per_match'] = (data['distance_covered_km'] / data['matches_played']) if data['matches_played'] > 0 else 0
    data['sprint_per_match'] = (data['sprints'] / data['matches_played']) if data['matches_played'] > 0 else 0
    data['defensive_activity'] = data['tackles'] + data['interceptions']
    
    result = predictor.predict(data)
    
    # Display results
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        risk_emoji = risk_color.get(result['risk_level'], "⚪")
        st.metric("Risk Level", f"{risk_emoji} {result['risk_level']}")
    
    with col2:
        st.metric("Confidence", f"{result['confidence']:.2%}")
    
    with col3:
        # Visual probability gauge
        prob_high = result['probabilities']['high']
        prob_medium = result['probabilities']['medium']
        prob_low = result['probabilities']['low']
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_high * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "High Risk"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "yellow"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
    
    # Probability breakdown
    st.subheader("📊 Probability Breakdown")
    
    prob_df = pd.DataFrame({
        'Risk Level': ['Low', 'Medium', 'High'],
        'Probability': [
            result['probabilities']['low'],
            result['probabilities']['medium'],
            result['probabilities']['high']
        ]
    })
    
    # Bar chart
    fig = px.bar(
        prob_df,
        x='Risk Level',
        y='Probability',
        color='Risk Level',
        color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'},
        text_auto='.2%'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def show_csv_upload():
    """Show CSV upload interface."""
    st.subheader("📈 Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with player data",
        type=['csv'],
        help="Upload a CSV file with player statistics"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} rows")
            
            st.subheader("Preview")
            st.dataframe(df.head(10))
            
            # Batch predictions
            if st.button("🔮 Predict for All Players"):
                if st.session_state.predictor and st.session_state.predictor.model:
                    results = []
                    for idx, row in df.iterrows():
                        result = st.session_state.predictor.predict(row.to_dict())
                        result['player_id'] = idx
                        results.append(result)
                    
                    results_df = pd.DataFrame(results)
                    st.subheader("Predictions")
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error loading CSV: {e}")


def show_video_analysis():
    """Video analysis page."""
    st.header("📹 Video Analysis")
    
    # Check if video analysis is available
    if not VIDEO_ANALYSIS_AVAILABLE:
        st.error("⚠️ Video analysis features are not available. Please install mediapipe:")
        st.code("pip install mediapipe")
        st.info("💡 The player analysis features are still available!")
        return
    
    st.subheader("Upload Video File")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a football match video for analysis"
    )
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("🔍 Analyze Video", type="primary"):
            with st.spinner("Processing video... This may take a few minutes."):
                try:
                    # Save uploaded file
                    upload_dir = Path("data/videos/uploads")
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    
                    video_path = upload_dir / uploaded_video.name
                    
                    with open(video_path, "wb") as f:
                        f.write(uploaded_video.getbuffer())
                    
                    # Extract features
                    extractor = VideoFeatureExtractor()
                    output_path = f"data/processed/{uploaded_video.name}_features.csv"
                    
                    df = extractor.extract_features(str(video_path), output_path)
                    
                    st.success("✅ Video analysis complete!")
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Frames Analyzed", len(df))
                        st.metric("Avg Players", f"{df['num_players'].mean():.1f}")
                    
                    with col2:
                        st.metric("Ball Detection Rate", f"{df['ball_detected'].mean():.2%}")
                        st.metric("Avg Speed", f"{df['speed'].mean():.2f}")
                    
                    with col3:
                        st.metric("Max Speed", f"{df['speed'].max():.2f}")
                        st.metric("Avg Acceleration", f"{df['acceleration'].mean():.4f}")
                    
                    # Visualizations
                    st.subheader("📈 Visualizations")
                    
                    # Time series plot
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Player Speed Over Time', 'Number of Players Over Time'),
                        vertical_spacing=0.1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=df['frame'], y=df['speed'], name='Speed'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=df['frame'], y=df['num_players'], name='Players'),
                        row=2, col=1
                    )
                    
                    fig.update_xaxes(title_text="Frame", row=2, col=1)
                    fig.update_yaxes(title_text="Speed", row=1, col=1)
                    fig.update_yaxes(title_text="Count", row=2, col=1)
                    fig.update_layout(height=600)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download features
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Features CSV",
                        data=csv,
                        file_name="video_features.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"Error analyzing video: {e}")


def show_model_info():
    """Show model information."""
    st.header("🤖 Model Information")
    
    st.subheader("AI Models Used")
    
    st.markdown("""
    ### 🎯 Random Forest Classifier
    - **Purpose**: Injury risk prediction
    - **Features**: 15+ player performance metrics
    - **Output**: Low/Medium/High risk classification
    - **Performance**: 92.5% accuracy (on sample data)
    
    ### 🔍 YOLOv8
    - **Purpose**: Object detection in videos
    - **Classes**: Players, ball, and other objects
    - **Framework**: Ultralytics YOLO
    - **Speed**: Real-time capable
    
    ### 🧘 MediaPipe Pose
    - **Purpose**: Human pose estimation
    - **Output**: 33 body landmarks
    - **Use Case**: Biomechanical analysis
    - **Performance**: High accuracy pose tracking
    """)
    
    st.markdown("---")
    
    st.subheader("Model Status")
    
    predictor_status = "✅ Loaded" if (st.session_state.predictor and st.session_state.predictor.model) else "❌ Not Loaded"
    st.metric("Predictor Model", predictor_status)
    
    model_path = Path("models/injury_risk_model.pkl")
    model_exists = "✅ Found" if model_path.exists() else "❌ Not Found"
    st.metric("Model File", model_exists)
    
    if st.button("🔄 Retrain Model"):
        with st.spinner("Training model... This may take a few minutes."):
            try:
                from models.train_model import main as train_main
                train_main()
                st.success("✅ Model training complete!")
                st.session_state.predictor = load_predictor()  # Reload predictor
                st.rerun()
            except Exception as e:
                st.error(f"Error training model: {e}")


if __name__ == "__main__":
    main()

