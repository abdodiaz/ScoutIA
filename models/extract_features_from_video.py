"""
Video Feature Extraction Module for ScoutIA Pro
Integrates YOLOv8 for player/ball detection and MediaPipe for pose estimation.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from ultralytics import YOLO
import mediapipe as mp
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    """Extracts features from football videos using YOLO and MediaPipe."""
    
    def __init__(self, yolo_model_path: str = "YOLO/yolov8n.pt"):
        """
        Initialize feature extractor with YOLO and MediaPipe models.
        
        Args:
            yolo_model_path: Path to YOLO model weights
        """
        # Initialize YOLO
        self.yolo_model_path = Path(yolo_model_path)
        if self.yolo_model_path.exists():
            self.yolo_model = YOLO(str(self.yolo_model_path))
            logger.info(f"Loaded YOLO model from {yolo_model_path}")
        else:
            logger.warning(f"YOLO model not found at {yolo_model_path}. Using default.")
            self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        logger.info("Initialized MediaPipe Pose")
        
        # COCO class names (YOLO uses COCO)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def detect_objects(self, frame: np.ndarray) -> list:
        """
        Detect objects in a frame using YOLO.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detected objects with bounding boxes and classes
        """
        results = self.yolo_model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates, confidence, and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'confidence': float(conf),
                    'class': cls,
                    'class_name': self.class_names[cls] if cls < len(self.class_names) else 'unknown'
                })
        
        return detections
    
    def extract_pose_keypoints(self, frame: np.ndarray) -> dict or None:
        """
        Extract pose keypoints using MediaPipe.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with pose keypoints or None
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract relevant keypoints
        landmarks = results.pose_landmarks.landmark
        
        # Calculate angles between joints
        angles = self._calculate_joint_angles(landmarks)
        
        # Extract key joint positions
        keypoints = {
            'left_shoulder': {'x': landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             'y': landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y},
            'right_shoulder': {'x': landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              'y': landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y},
            'left_hip': {'x': landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                        'y': landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y},
            'right_hip': {'x': landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                         'y': landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y},
            'left_knee': {'x': landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                         'y': landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y},
            'right_knee': {'x': landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x,
                          'y': landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y},
            'left_ankle': {'x': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                          'y': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y},
            'right_ankle': {'x': landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                           'y': landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y},
        }
        
        return {
            'keypoints': keypoints,
            'angles': angles
        }
    
    def _calculate_joint_angles(self, landmarks) -> dict:
        """Calculate angles between joints for biomechanical analysis."""
        def angle_between_points(p1, p2, p3):
            """Calculate angle at p2 between p1-p2-p3."""
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
        
        angles = {}
        
        # Left leg angles
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        angles['left_knee_angle'] = angle_between_points(left_hip, left_knee, left_ankle)
        
        # Right leg angles
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        angles['right_knee_angle'] = angle_between_points(right_hip, right_knee, right_ankle)
        
        return angles
    
    def calculate_motion_metrics(self, positions: list) -> dict:
        """
        Calculate motion metrics from position history.
        
        Args:
            positions: List of (x, y) position tuples
            
        Returns:
            Dictionary with speed, acceleration, etc.
        """
        if len(positions) < 2:
            return {'speed': 0, 'acceleration': 0}
        
        positions = np.array(positions)
        
        # Calculate distances between consecutive positions
        distances = np.diff(positions, axis=0)
        distances = np.linalg.norm(distances, axis=1)
        
        # Average speed (pixels per frame)
        avg_speed = np.mean(distances) if len(distances) > 0 else 0
        
        # Acceleration
        if len(distances) > 1:
            speed_changes = np.diff(distances)
            avg_acceleration = np.mean(speed_changes) if len(speed_changes) > 0 else 0
        else:
            avg_acceleration = 0
        
        return {
            'speed': float(avg_speed),
            'acceleration': float(avg_acceleration),
            'total_distance': float(np.sum(distances))
        }
    
    def extract_features(self, video_path: str, 
                        output_path: str = "data/processed/video_features.csv",
                        sample_rate: int = 10) -> pd.DataFrame:
        """
        Extract features from a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save extracted features
            sample_rate: Extract features every N frames
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return pd.DataFrame()
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Tracking variables
        frame_features = []
        frame_idx = 0
        last_positions = {}  # Track positions for each player
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on sample rate
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue
            
            frame_features_row = {'frame': frame_idx, 'timestamp': frame_idx / fps}
            
            # Detect objects
            detections = self.detect_objects(frame)
            
            # Count players and ball
            players = [d for d in detections if d['class_name'] == 'person']
            ball = [d for d in detections if d['class_name'] == 'sports ball']
            
            frame_features_row['num_players'] = len(players)
            frame_features_row['ball_detected'] = 1 if len(ball) > 0 else 0
            
            # Average player position
            if players:
                avg_x = np.mean([p['x1'] + (p['x2'] - p['x1']) / 2 for p in players])
                avg_y = np.mean([p['y1'] + (p['y2'] - p['y1']) / 2 for p in players])
                frame_features_row['avg_player_x'] = avg_x
                frame_features_row['avg_player_y'] = avg_y
                
                # Update positions for speed calculation
                player_id = 0  # Simplified: track first player
                if player_id not in last_positions:
                    last_positions[player_id] = []
                last_positions[player_id].append((avg_x, avg_y))
                
                # Keep only last 10 positions
                if len(last_positions[player_id]) > 10:
                    last_positions[player_id].pop(0)
                
                # Calculate motion metrics
                motion_metrics = self.calculate_motion_metrics(last_positions[player_id])
                frame_features_row['speed'] = motion_metrics['speed']
                frame_features_row['acceleration'] = motion_metrics['acceleration']
            else:
                frame_features_row['avg_player_x'] = 0
                frame_features_row['avg_player_y'] = 0
                frame_features_row['speed'] = 0
                frame_features_row['acceleration'] = 0
            
            # Extract pose keypoints from first player
            if players and len(players) > 0:
                # Crop to first player for pose estimation
                p = players[0]
                x1, y1, x2, y2 = int(p['x1']), int(p['y1']), int(p['x2']), int(p['y2'])
                # Add padding
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)
                
                player_crop = frame[y1:y2, x1:x2]
                
                pose_data = self.extract_pose_keypoints(player_crop)
                
                if pose_data:
                    # Add knee angles
                    frame_features_row['left_knee_angle'] = pose_data['angles'].get('left_knee_angle', 0)
                    frame_features_row['right_knee_angle'] = pose_data['angles'].get('right_knee_angle', 0)
                else:
                    frame_features_row['left_knee_angle'] = 0
                    frame_features_row['right_knee_angle'] = 0
            else:
                frame_features_row['left_knee_angle'] = 0
                frame_features_row['right_knee_angle'] = 0
            
            frame_features.append(frame_features_row)
            frame_idx += 1
            
            # Progress update
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Processed {frame_idx}/{total_frames} frames ({elapsed:.1f}s)")
        
        cap.release()
        
        # Create DataFrame
        df = pd.DataFrame(frame_features)
        
        # Save to CSV
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        elapsed = time.time() - start_time
        logger.info(f"Feature extraction complete. Saved {len(df)} rows to {output_path} in {elapsed:.1f}s")
        
        return df


def main():
    """Example usage of video feature extraction."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_features_from_video.py <video_path> [output_path]")
        return
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/video_features.csv"
    
    # Initialize extractor
    extractor = VideoFeatureExtractor()
    
    # Extract features
    df = extractor.extract_features(video_path, output_path)
    
    print("\nExtracted Features Summary:")
    print(df.describe())
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()

