"""Hand tracking module using MediaPipe."""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path



class LandmarkSmoother:
    """Applies Exponential Moving Average (EMA) to landmarks."""
    
    def __init__(self, alpha: float = 0.7, jump_threshold: float = 0.1):
        self.alpha = alpha
        self.jump_threshold = jump_threshold
        self.prev_landmarks = None
        
    def update(self, current_landmarks):
        """Smooth landmarks using EMA."""
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return current_landmarks
            
        smoothed = []
        for i, curr in enumerate(current_landmarks):
            prev = self.prev_landmarks[i]
            
            # Calculate distance
            dx = curr.x - prev.x
            dy = curr.y - prev.y
            dz = curr.z - prev.z
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Selective EMA: Reset if jump is too large (tracking glitch)
            if dist > self.jump_threshold:
                new_point = curr
            else:
                # Smooth
                new_x = self.alpha * curr.x + (1 - self.alpha) * prev.x
                new_y = self.alpha * curr.y + (1 - self.alpha) * prev.y
                new_z = self.alpha * curr.z + (1 - self.alpha) * prev.z
                
                # Create a simple object with x,y,z attributes to mimic MediaPipe landmark
                new_point = type(curr)(x=new_x, y=new_y, z=new_z, 
                                     visibility=curr.visibility, 
                                     presence=curr.presence)
            
            smoothed.append(new_point)
            
        self.prev_landmarks = smoothed
        return smoothed


class HandTracker:
    """Tracks hand landmarks using MediaPipe Hand Landmarker."""
    
    def __init__(self, model_path: str = None):
        """Initialize the hand tracker.
        
        Args:
            model_path: Path to hand_landmarker.task model file.
        """
        if model_path is None:
            # Look for model in project root
            model_path = Path(__file__).parent.parent / "hand_landmarker.task"
        
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)
        self.frame_timestamp_ms = 0
        
        self.smoother = LandmarkSmoother(alpha=0.6, jump_threshold=0.1)
        self.last_sane_landmarks = None
        self.last_sane_timestamp = 0
        
        if not self.cap.isOpened():
            raise RuntimeError("❌ Webcam not detected")
        
        print("[OK] Hand tracker initialized")
    
    def process(self):
        """Capture frame and detect hand landmarks.
        
        Returns:
            tuple: (frame, landmarks) where landmarks is list of 21 points or None
        """
        success, frame = self.cap.read()
        if not success:
            return None, None
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect
        self.frame_timestamp_ms += 33  # ~30 FPS
        result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        
        if result.hand_landmarks:
            raw_landmarks = result.hand_landmarks[0]
            
            # Apply smoothing and sanity checks
            if self._is_velocity_sane(raw_landmarks, self.frame_timestamp_ms):
                landmarks = self.smoother.update(raw_landmarks)
            else:
                print("⚠️ Skipped frame: velocity too high")
                return frame, None
        else:
             landmarks = None
        
        return frame, landmarks

    def _is_velocity_sane(self, landmarks, timestamp_ms: int) -> bool:
        """Check if hand movement is within human speed limits."""
        if self.last_sane_landmarks is None:
            self.last_sane_landmarks = landmarks
            self.last_sane_timestamp = timestamp_ms
            return True
            
        dt = (timestamp_ms - self.last_sane_timestamp) / 1000.0
        if dt <= 0: return True
        
        # Check wrist velocity
        wrist = landmarks[0]
        prev_wrist = self.last_sane_landmarks[0]
        
        dx = wrist.x - prev_wrist.x
        dy = wrist.y - prev_wrist.y
        dz = wrist.z - prev_wrist.z
        
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        speed = dist / dt  # units per second
        
        # Max sane speed (screen width per second)
        MAX_SPEED = 5.0 # Increased from 2.0 to prevent blocking normal usage 
        
        if speed > MAX_SPEED:
            return False
            
        self.last_sane_landmarks = landmarks
        self.last_sane_timestamp = timestamp_ms
        return True


    
    def get_landmark_position(self, landmarks, index: int):
        """Get normalized (x, y, z) position of a landmark.
        
        Args:
            landmarks: MediaPipe hand landmarks
            index: Landmark index (0-20)
            
        Returns:
            tuple: (x, y, z) normalized coordinates
        """
        if landmarks is None:
            return None
        
        lm = landmarks[index]
        return (lm.x, lm.y, lm.z)
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame.
        
        Args:
            frame: OpenCV BGR frame
            landmarks: MediaPipe hand landmarks
            
        Returns:
            frame: Frame with landmarks drawn
        """
        if landmarks is None:
            return frame
        
        h, w, _ = frame.shape
        
        # Hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Convert to pixel coordinates
        points = []
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
        
        # Draw connections
        for start_idx, end_idx in connections:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        
        # Draw points
        for point in points:
            cv2.circle(frame, point, 5, (255, 0, 0), -1)
        
        return frame
    
    def release(self):
        """Release webcam resources."""
        self.cap.release()
        print("✅ Hand tracker released")
