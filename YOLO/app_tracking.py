import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

st.set_page_config(page_title="Football Team Analysis", layout="wide")
st.title("⚽ Analyse complète des équipes - YOLOv8 + Tracking + Stats")

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")

# --- Upload vidéo ---
uploaded_file = st.file_uploader("📤 Upload une vidéo de match", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file is None:
    st.stop()

# Sauvegarde temporaire
tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded_file.read())
video_path = tfile.name

# Conteneur Streamlit
stframe = st.empty()
st.info("⏳ Initialisation du tracking et analyse des couleurs...")

# --- Variables globales ---
player_teams = {}
player_positions = {}
colors_buffer = []
kmeans = None
results_data = []

# --- Lecture vidéo ---
cap = cv2.VideoCapture(video_path)
frame_id = 0
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    results = model.track(source=frame, tracker="bytetrack.yaml", persist=True, verbose=False)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        # Collecte des couleurs sur premières frames
        if frame_id < 60:
            for i, track_id in enumerate(ids):
                x1, y1, x2, y2 = map(int, boxes[i])
                roi = frame[int(y1 + (y2 - y1) / 3):int(y1 + (y2 - y1) * 2 / 3), int(x1):int(x2)]
                if roi.size > 0:
                    avg_color = np.mean(roi.reshape(-1, 3), axis=0)
                    colors_buffer.append(avg_color)
            if frame_id == 60 and len(colors_buffer) > 2:
                kmeans = KMeans(n_clusters=2, random_state=42).fit(colors_buffer)
                st.success("🎨 Couleurs d'équipes détectées automatiquement")

        # Si équipes prêtes
        if kmeans is not None:
            for i, track_id in enumerate(ids):
                x1, y1, x2, y2 = map(int, boxes[i])
                roi = frame[int(y1 + (y2 - y1) / 3):int(y1 + (y2 - y1) * 2 / 3), int(x1):int(x2)]
                if roi.size == 0:
                    continue
                avg_color = np.mean(roi.reshape(-1, 3), axis=0).reshape(1, -1)

                if track_id not in player_teams:
                    label = kmeans.predict(avg_color)[0]
                    player_teams[track_id] = label

                team = player_teams[track_id]
                color_box = (255, 0, 0) if team == 0 else (0, 0, 255)

                # Position du centre
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Sauvegarde position pour calculs
                if track_id not in player_positions:
                    player_positions[track_id] = []
                player_positions[track_id].append((frame_id, cx, cy, team))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(frame, f"ID:{int(track_id)} T{team+1}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)

    stframe.image(frame, channels="BGR", use_container_width=True)

cap.release()
st.success("✅ Tracking terminé ! Calcul des statistiques...")

# --- Sauvegarde CSV ---
rows = []
for pid, positions in player_positions.items():
    for (frame, x, y, team) in positions:
        rows.append([frame, pid, team + 1, x, y])
df = pd.DataFrame(rows, columns=["frame", "player_id", "team", "x", "y"])
csv_path = "player_tracking_data.csv"
df.to_csv(csv_path, index=False)
st.download_button("⬇️ Télécharger le CSV", data=open(csv_path, "rb"), file_name=csv_path)

# --- Analyse statistique ---
st.header("📊 Statistiques par joueur")

def compute_stats(df):
    stats = []
    for pid in df["player_id"].unique():
        player_df = df[df["player_id"] == pid].sort_values("frame")
        team = player_df["team"].iloc[0]
        dist = 0
        for i in range(1, len(player_df)):
            dx = player_df.iloc[i]["x"] - player_df.iloc[i-1]["x"]
            dy = player_df.iloc[i]["y"] - player_df.iloc[i-1]["y"]
            dist += math.sqrt(dx**2 + dy**2)
        time_s = len(player_df) / fps
        speed_avg = dist / time_s if time_s > 0 else 0
        stats.append([pid, team, round(dist,2), round(speed_avg,2)])
    return pd.DataFrame(stats, columns=["Player ID", "Team", "Distance(px)", "Avg Speed(px/s)"])

player_stats = compute_stats(df)
st.dataframe(player_stats)

# --- Heatmap par équipe ---
st.header("🔥 Heatmap des positions par équipe")

for team_id in sorted(df["team"].unique()):
    team_df = df[df["team"] == team_id]
    plt.figure(figsize=(6,4))
    sns.kdeplot(x=team_df["x"], y=team_df["y"], fill=True, thresh=0.05)
    plt.title(f"Heatmap Équipe {team_id}")
    plt.gca().invert_yaxis()
    st.pyplot(plt)
