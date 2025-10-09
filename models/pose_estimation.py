"""pose_estimation.py
Exemple d'utilisation de MediaPipe Pose pour extraire keypoints.
"""
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_pose(video_path: str, out_csv: str = None):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    frame_idx = 0
    rows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(image_rgb)
        if res.pose_landmarks:
            # collect some example keypoints (x,y) for hip, knee, ankle (left/right)
            lm = res.pose_landmarks.landmark
            row = {
                'frame': frame_idx,
                'left_hip_x': lm[mp_pose.PoseLandmark.LEFT_HIP].x,
                'left_hip_y': lm[mp_pose.PoseLandmark.LEFT_HIP].y,
                'left_knee_x': lm[mp_pose.PoseLandmark.LEFT_KNEE].x,
                'left_knee_y': lm[mp_pose.PoseLandmark.LEFT_KNEE].y,
            }
            rows.append(row)
    cap.release()
    import pandas as pd
    if out_csv:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: python pose_estimation.py path/to/video.mp4 [out.csv]')
    else:
        extract_pose(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
