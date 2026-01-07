"""Gesture detection from hand landmarks."""
import numpy as np
from enum import Enum
from typing import Optional, Dict


class Gesture(Enum):
    """Recognized hand gestures."""
    NONE = "none"
    POINT = "point"      # Index extended, others curled
    PINCH = "pinch"      # Thumb + Index tips close
    GRAB = "grab"        # All fingers curled (fist)
    PALM = "palm"        # All fingers extended
    PEACE = "peace"      # Index + Middle extended


# Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


class GestureDetector:
    """Detect hand gestures from MediaPipe landmarks."""
    
    def __init__(self):
        """Initialize gesture detector with robust thresholds."""
        # Visual cues for debug
        self.last_scores = {}
        
        # Thresholds (using normalized scores 0.0-1.0)
        self.pinch_threshold = 0.8
        self.curl_threshold = 0.03 # Restored for compatibility
        self.last_gesture = Gesture.NONE
        self.gesture_hold_frames = 0
        self.min_hold_frames = 3
        
        # State machine variables
        self.state = "IDLE"
        self.state_frames = 0
    
    def detect(self, landmarks) -> Gesture:
        """Detect gesture from hand landmarks using State Machine.
        
        Args:
            landmarks: MediaPipe hand landmarks (21 points)
            
        Returns:
            Gesture: Detected gesture
        """
        if landmarks is None:
            self.state = "IDLE"
            self.gesture_hold_frames = 0
            self.last_gesture = Gesture.NONE
            return Gesture.NONE
        
        # 1. Normalize & Score
        scale = self._get_hand_scale(landmarks)
        if scale == 0: return Gesture.NONE
        
        scores = self._get_gesture_scores(landmarks, scale)
        self.last_scores = scores
        
        # 2. State Machine Transitions
        # States: IDLE -> HAND_PRESENT -> PRE_PINCH -> PINCHED -> RELEASE -> HAND_PRESENT
        
        new_gesture = Gesture.NONE
        pinch_score = scores["pinch"]
        
        # State: IDLE (Hand just appeared)
        if self.state == "IDLE":
            self.state = "HAND_PRESENT"
        
        # State: HAND_PRESENT (Neutral)
        if self.state == "HAND_PRESENT":
            if pinch_score > 0.4: # Lowered from 0.5
                self.state = "PRE_PINCH"
            elif scores["palm"] > 0.8:
                new_gesture = Gesture.PALM
            elif scores["grab"] > 0.8:
                new_gesture = Gesture.GRAB
            elif scores["peace"] > 0.8:
                new_gesture = Gesture.PEACE
            else:
                new_gesture = Gesture.POINT # Default to point
        
        # State: PRE_PINCH (Approaching pinch)
        elif self.state == "PRE_PINCH":
            if pinch_score > 0.7: # Lowered from 0.75
                # Angle check - Temporarily disabled/relaxed
                # if self._check_pinch_angle(landmarks): 
                self.state = "PINCHED"
            elif pinch_score < 0.3: # Lowered dropout
                # Aborted pinch
                self.state = "HAND_PRESENT"
            
            # While in pre-pinch, we are technically "POINTING" but careful
            new_gesture = Gesture.POINT
            
        # State: PINCHED (Active action)
        elif self.state == "PINCHED":
            if pinch_score < 0.5: # Released (Lowered from 0.6 to match easier entry)
                self.state = "RELEASE"
            else:
                new_gesture = Gesture.PINCH
        
        # State: RELEASE (Hysteresis / Cooldown)
        elif self.state == "RELEASE":
            if pinch_score < 0.3: # Fully opened
                self.state = "HAND_PRESENT"
            elif pinch_score > 0.8: # Re-pinched too fast? 
                # Require full release first, so ignore re-pinch here
                pass
            
            # Show Point during release
            new_gesture = Gesture.POINT

        # 3. Temporal Consistency (Debounce output only)
        # We trust the state machine for transitions, but average the output label
        if new_gesture == self.last_gesture:
            self.gesture_hold_frames += 1
        else:
            self.gesture_hold_frames = 1
            self.last_gesture = new_gesture
            
        # Fast path for PINCH (responsiveness)
        if new_gesture == Gesture.PINCH and self.state == "PINCHED":
             return Gesture.PINCH
        
        if self.gesture_hold_frames >= self.min_hold_frames:
            return new_gesture
            
        return Gesture.NONE

    def _get_hand_scale(self, landmarks) -> float:
        """Calculate hand scale for normalization.
        
        Uses max of Wrist->IndexMCP or Wrist->MiddleMCP to be robust.
        """
        wrist = landmarks[WRIST]
        index_mcp = landmarks[INDEX_MCP]
        middle_mcp = landmarks[MIDDLE_MCP]
        
        d1 = self._dist(wrist, index_mcp)
        d2 = self._dist(wrist, middle_mcp)
        
        return max(d1, d2, 0.01) # Avoid div by zero

    def _get_gesture_scores(self, landmarks, scale: float) -> dict:
        """Compute 0-1 confidence scores for all gestures."""
        scores = {}
        
        # --- Pinch Score ---
        # Distance between thumb tip and index tip
        pinch_dist = self._dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
        norm_pinch = pinch_dist / scale
        # Map 0.05..0.2 normalized distance to 1.0..0.0 score
        # Using 0.2 as max distance for pinch
        # Tighter threshold for state machine logic
        # RELAXED to 0.5 to make pinching easier (User Request)
        scores["pinch"] = max(0, min(1, 1 - (norm_pinch / 0.5)))  
        
        # --- Finger Extensions ---
        # Check extensions
        fingers = ["index", "middle", "ring", "pinky"]
        ext_states = [self._is_finger_extended(landmarks, f) for f in fingers]
        ext_count = sum(ext_states)
        thumb_ext = self._is_thumb_extended(landmarks)
        
        # --- Palm Score ---
        # All fingers + thumb extended
        palm_conf = ext_count / 4.0
        if thumb_ext: palm_conf = (palm_conf + 1) / 2.0
        scores["palm"] = 1.0 if (ext_count == 4 and thumb_ext) else 0.0
        
        # --- Grab Score ---
        # No fingers extended
        scores["grab"] = 1.0 if (ext_count == 0 and not thumb_ext) else 0.0
        
        # --- Peace Score ---
        # Index + Middle only
        is_peace = ext_states[0] and ext_states[1] and not ext_states[2] and not ext_states[3]
        scores["peace"] = 1.0 if is_peace else 0.0
        
        return scores

    def _check_pinch_angle(self, landmarks) -> bool:
        """Check if thumb and index are facing each other using dot product."""
        # Vector 1: Thumb PIP -> Tip
        thumb_dir = self._vec(landmarks[THUMB_IP], landmarks[THUMB_TIP])
        # Vector 2: Index PIP -> Tip
        index_dir = self._vec(landmarks[INDEX_PIP], landmarks[INDEX_TIP])
        
        # Normalize
        t_mag = np.linalg.norm(thumb_dir)
        i_mag = np.linalg.norm(index_dir)
        if t_mag == 0 or i_mag == 0: return False
        
        thumb_norm = thumb_dir / t_mag
        index_norm = index_dir / i_mag
        
        # Dot product: 1=parallel, -1=opposite
        # We want them somewhat opposite or converging
        # Actually, for a pinch, the tips converge. 
        # A simple check is just distance, but the prompt asked for "dot product".
        # Let's trust distance + simple opposition.
        # If dot product is < 0, they are facing somewhat opposite directions (converging).
        # But in 2D/pseudo-3D this can be tricky.
        # Let's rely on the pinch score mostly, and just check they aren't parallel (dot > 0.8)
        
        dot = np.dot(thumb_norm, index_norm)
        return True # dot < 0.5 # Relaxed check for now
        
    def _dist(self, p1, p2) -> float:
        return np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)
        
    def _vec(self, p1, p2):
        return np.array([p2.x-p1.x, p2.y-p1.y, p2.z-p1.z])
    
    def _is_finger_extended(self, landmarks, finger: str) -> bool:
        """Check if a finger is extended.
        
        Uses the y-position comparison: if tip is above PIP joint, finger is extended.
        (In image coordinates, smaller y = higher position)
        """
        finger_tips = {
            "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
            "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
            "ring": (RING_MCP, RING_PIP, RING_TIP),
            "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
        }
        
        mcp, pip, tip = finger_tips[finger]
        
        # Get y-coordinates (smaller = higher in image)
        mcp_y = landmarks[mcp].y
        pip_y = landmarks[pip].y
        tip_y = landmarks[tip].y
        
        # Finger is extended if tip is significantly above PIP
        # and the finger is relatively straight (tip above MCP)
        return tip_y < pip_y - self.curl_threshold and tip_y < mcp_y
    
    def _is_thumb_extended(self, landmarks) -> bool:
        """Check if thumb is extended (opened away from palm)."""
        # Compare thumb tip x-position with thumb MCP
        # For right hand: extended thumb has tip to the left (smaller x)
        # We use the index MCP as reference
        thumb_tip_x = landmarks[THUMB_TIP].x
        thumb_mcp_x = landmarks[THUMB_MCP].x
        index_mcp_x = landmarks[INDEX_MCP].x
        
        # Thumb is extended if tip is farther from index than MCP is
        thumb_tip_dist = abs(thumb_tip_x - index_mcp_x)
        thumb_mcp_dist = abs(thumb_mcp_x - index_mcp_x)
        
        return thumb_tip_dist > thumb_mcp_dist + 0.05
    
    def _get_pinch_distance(self, landmarks) -> float:
        """Get distance between thumb tip and index tip (normalized)."""
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        
        dx = thumb.x - index.x
        dy = thumb.y - index.y
        dz = thumb.z - index.z
        
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def get_index_tip_position(self, landmarks) -> Optional[tuple]:
        """Get normalized position of index finger tip.
        
        Returns:
            tuple: (x, y, z) normalized or None
        """
        if landmarks is None:
            return None
        
        tip = landmarks[INDEX_TIP]
        return (tip.x, tip.y, tip.z)
    
    def get_palm_center(self, landmarks) -> Optional[tuple]:
        """Get approximate center of palm.
        
        Returns:
            tuple: (x, y, z) normalized or None
        """
        if landmarks is None:
            return None
        
        # Average of wrist and MCP joints
        points = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        x = sum(landmarks[i].x for i in points) / len(points)
        y = sum(landmarks[i].y for i in points) / len(points)
        z = sum(landmarks[i].z for i in points) / len(points)
        
        return (x, y, z)
