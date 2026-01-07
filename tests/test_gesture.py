
import sys
import os
import unittest
from unittest.mock import MagicMock
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gesture_detector import GestureDetector, Gesture
from src.gesture_detector import THUMB_TIP, INDEX_TIP, WRIST, INDEX_MCP, MIDDLE_MCP

# Mock Landmark class
class MockLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

def create_mock_hand(pinch_dist=1.0, scale=0.2):
    """Create a mock list of 21 landmarks."""
    landmarks = [MockLandmark(0,0,0) for _ in range(21)]
    
    # Setup Wrist and MCPs for scale
    landmarks[WRIST] = MockLandmark(0.5, 0.8)
    landmarks[INDEX_MCP] = MockLandmark(0.5, 0.6) # Dist = 0.2
    landmarks[MIDDLE_MCP] = MockLandmark(0.55, 0.6)
    
    # Setup Pinch
    # Thumb Tip at (0.5, 0.4)
    # Index Tip at (0.5 + pinch_dist, 0.4)
    landmarks[THUMB_TIP] = MockLandmark(0.5, 0.4)
    landmarks[INDEX_TIP] = MockLandmark(0.5 + pinch_dist, 0.4)
    
    # Needed for Angle Check (PIP -> Tip vectors)
    # Thumb: IP -> Tip
    landmarks[3] = MockLandmark(0.4, 0.4) # Left of tip -> Vector (0.1, 0, 0)
    # Index: PIP -> Tip
    landmarks[6] = MockLandmark(0.6 + pinch_dist, 0.4) # Right of tip -> Vector (-0.1, 0, 0)
    # Vectors pointing towards each other -> Dot product = -1 (approx)
    
    return landmarks

class TestGestureDetector(unittest.TestCase):
    def setUp(self):
        self.detector = GestureDetector()
        
    def test_state_machine_cycle(self):
        print("\n--- Testing State Machine Cycle ---")
        
        # 1. Idle
        print("1. Testing IDLE -> HAND_PRESENT")
        self.detector.detect(None)
        self.assertEqual(self.detector.state, "IDLE")
        
        hand_open = create_mock_hand(pinch_dist=0.15) # Normalized ~ 0.15/0.2 = 0.75 distance -> Score low
        self.detector.detect(hand_open)
        self.assertEqual(self.detector.state, "HAND_PRESENT")
        
        # 2. Pre-Pinch
        print("2. Testing HAND_PRESENT -> PRE_PINCH")
        # Pinch score = 1 - (dist / scale / 0.25). 
        # Target score > 0.6 for Pre-pinch.
        # Let scale ~ 0.2. 
        # dist = 0.05 -> norm = 0.25 -> score = 0 -> No.
        # dist = 0.01 -> norm = 0.05 -> score = 0.8 -> Yes.
        
        hand_pre = create_mock_hand(pinch_dist=0.04) # 0.04 / 0.2 = 0.2. Score = 1 - (0.2/0.25) = 0.2. Wait.
        # Check math: score = max(0, min(1, 1 - (norm_pinch / 0.25)))
        # To get score > 0.6: 1 - x/0.25 > 0.6 => x/0.25 < 0.4 => x < 0.1.
        # norm_pinch < 0.1.
        # dist < 0.1 * 0.2 = 0.02.
        
        hand_pre = create_mock_hand(pinch_dist=0.015) 
        # 0.015 / 0.2 = 0.075.
        # Score = 1 - (0.075/0.25) = 1 - 0.3 = 0.7.
        # 0.7 > 0.6. Should be PRE_PINCH.
        
        self.detector.detect(hand_pre)
        print(f"State: {self.detector.state} (Score: {self.detector.last_scores['pinch']:.2f})")
        self.assertEqual(self.detector.state, "PRE_PINCH")
        
        # 3. Pinch
        print("3. Testing PRE_PINCH -> PINCHED")
        hand_pinch = create_mock_hand(pinch_dist=0.001) # Very close
        self.detector.detect(hand_pinch)
        print(f"State: {self.detector.state}")
        self.assertEqual(self.detector.state, "PINCHED")
        
        # 4. Hold Pinch
        print("4. Testing Hold PINCHED")
        for _ in range(5):
            self.detector.detect(hand_pinch)
        self.assertEqual(self.detector.state, "PINCHED")
        
        # 5. Release
        print("5. Testing PINCHED -> RELEASE")
        # Score < 0.6.
        # 1 - x/0.25 < 0.6 => x > 0.1.
        # dist > 0.02.
        hand_release = create_mock_hand(pinch_dist=0.03)
        self.detector.detect(hand_release)
        self.assertEqual(self.detector.state, "RELEASE")
        
        # 6. Back to Neutral
        print("6. Testing RELEASE -> HAND_PRESENT")
        # Score < 0.3.
        # 1 - x/0.25 < 0.3 => x > 0.175.
        # dist > 0.035.
        hand_far = create_mock_hand(pinch_dist=0.05)
        self.detector.detect(hand_far)
        self.assertEqual(self.detector.state, "HAND_PRESENT")
        
        print("✅ State Machine Cycle Verified")

if __name__ == "__main__":
    unittest.main()
