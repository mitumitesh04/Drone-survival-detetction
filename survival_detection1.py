import cv2
import numpy as np
import time
import threading
import queue
import logging
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import json
import os
from pathlib import Path
import math

# Try to import advanced libraries
try:
    import dlib
    DLIB_AVAILABLE = True
    print("✅ dlib available - Human distress detection enabled")
except ImportError:
    DLIB_AVAILABLE = False
    print("⚠️  dlib not available - Install with: pip install dlib")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO available - Disaster object detection enabled")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available - using simulated detection")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    print("✅ MediaPipe available")
except ImportError:
    MP_AVAILABLE = False
    print("⚠️  MediaPipe not available")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DISASTER SURVIVAL RELEVANT OBJECTS (Filtered from COCO for disaster response)
DISASTER_OBJECTS = {
    # 🚨 CRITICAL - Search & Rescue
    0: 'person',           # PRIMARY TARGET - survivors
    15: 'cat',            # Pet rescue
    16: 'dog',            # Pet rescue, search dogs
    17: 'horse',          # Livestock, transportation
    18: 'sheep',          # Livestock
    19: 'cow',            # Livestock
    
    # 🚗 VEHICLES - Transportation, Obstacles, Emergency Response
    1: 'bicycle',         # Personal transport
    2: 'car',            # Damaged vehicles, transportation
    3: 'motorcycle',      # Personal transport
    4: 'airplane',        # Rescue aircraft
    5: 'bus',            # Mass transport, shelter
    6: 'train',          # Mass transport
    7: 'truck',          # Emergency vehicles, supplies
    8: 'boat',           # Water rescue
    
    # 🏠 SHELTER & PROTECTION
    25: 'umbrella',       # Makeshift shelter, signaling
    56: 'chair',          # Furniture debris, makeshift tools
    57: 'couch',          # Furniture debris, shelter materials
    59: 'bed',            # Medical triage area, shelter
    60: 'dining table',   # Shelter materials, medical table
    
    # 📱 COMMUNICATION & SIGNALING
    67: 'cell phone',     # Critical communication device
    63: 'laptop',         # Communication, information
    62: 'tv',             # Information source
    33: 'kite',           # Signaling device
    32: 'sports ball',    # Signaling, makeshift tool
    
    # 🥤 SURVIVAL RESOURCES - Food & Water
    39: 'bottle',         # Water containers (CRITICAL)
    41: 'cup',            # Drinking vessels
    45: 'bowl',           # Food/water containers
    46: 'banana',         # Food source
    47: 'apple',          # Food source
    49: 'orange',         # Food source
    
    # 🎒 EQUIPMENT & SUPPLIES
    24: 'backpack',       # Survival gear storage
    28: 'suitcase',       # Supply storage
    43: 'knife',          # Essential survival tool
    76: 'scissors',       # Medical/utility tool
    73: 'book',           # Information, kindling
    
    # 🔥 HEAT & ENERGY (Inferred objects)
    # Note: COCO doesn't have fire, generators, etc. - these would need custom training
}

# DISASTER RESPONSE CATEGORIES
DISASTER_CATEGORIES = {
    'survivors': [0],  # person - HIGHEST PRIORITY
    'animals': [15, 16, 17, 18, 19],  # pets and livestock rescue
    'vehicles': [1, 2, 3, 4, 5, 6, 7, 8],  # transportation and obstacles
    'shelter': [25, 56, 57, 59, 60],  # shelter materials and debris
    'communication': [67, 63, 62, 33, 32],  # communication and signaling
    'survival_resources': [39, 41, 45, 46, 47, 49],  # food and water
    'equipment': [24, 28, 43, 76, 73],  # tools and supplies
}

# PRIORITY COLORS (Based on emergency importance)
PRIORITY_COLORS = {
    'survivors': (0, 255, 0),          # BRIGHT GREEN - Highest priority
    'animals': (0, 255, 255),          # YELLOW - High priority (pets/livestock)
    'vehicles': (255, 0, 0),           # RED - Medium priority (obstacles/transport)
    'shelter': (128, 0, 128),          # PURPLE - Medium priority
    'communication': (255, 165, 0),    # ORANGE - High priority
    'survival_resources': (0, 0, 255), # BLUE - Critical for survival
    'equipment': (255, 192, 203),      # PINK - Useful tools
    'unknown': (128, 128, 128)         # GRAY - Unknown objects
}

# EMERGENCY RESPONSE PRIORITIES
EMERGENCY_PRIORITIES = {
    'survivors': 10,        # MAXIMUM PRIORITY
    'communication': 9,     # Critical for coordination
    'survival_resources': 8, # Water/food essential
    'animals': 7,          # Pet/livestock rescue
    'vehicles': 6,         # Transportation/obstacles
    'equipment': 5,        # Useful tools
    'shelter': 4,          # Shelter materials
}

@dataclass
class DisasterDetection:
    """Disaster-specific object detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    category: str
    priority: int  # Emergency priority (1-10)
    object_id: int
    tracking_confidence: float = 1.0
    
    # Human-specific (survivor analysis)
    action: Optional[str] = None  # standing, sitting, lying_down
    distress_level: str = "unknown"  # low, medium, high, critical
    head_angle: float = 0.0
    body_angle: float = 0.0
    aspect_ratio: float = 0.0
    center_y_ratio: float = 0.0
    classification_confidence: float = 0.0
    needs_immediate_attention: bool = False
    
    # Survival resource analysis
    resource_condition: str = "unknown"  # good, damaged, unusable
    accessibility: str = "unknown"  # accessible, blocked, buried
    
    # Location context
    gps_coords: Optional[Tuple[float, float]] = None
    altitude: float = 0.0
    timestamp: float = 0.0

@dataclass
class SurvivorAnalysis:
    """Detailed survivor condition analysis"""
    person_id: int
    distress_indicators: List[str]  # lying_down, injured, trapped, etc.
    mobility_status: str  # mobile, limited, immobile
    estimated_condition: str  # good, injured, critical
    last_movement: float  # seconds since last detected movement
    requires_evacuation: bool
    medical_priority: int  # 1-5 scale

class EnhancedSurvivorPoseAnalyzer:
    """Enhanced pose classification using multi-feature analysis (from working code)"""
    
    def __init__(self):
        self.face_detector = None
        self.landmark_predictor = None
        self.pose_history = deque(maxlen=10)  # Temporal smoothing
        
        if DLIB_AVAILABLE:
            try:
                # Initialize dlib face detector
                self.face_detector = dlib.get_frontal_face_detector()
                
                # Try to load facial landmark predictor
                landmark_files = [
                    "shape_predictor_68_face_landmarks.dat",
                    "models/shape_predictor_68_face_landmarks.dat",
                    "/usr/local/share/dlib/shape_predictor_68_face_landmarks.dat"
                ]
                
                for landmark_file in landmark_files:
                    if os.path.exists(landmark_file):
                        self.landmark_predictor = dlib.shape_predictor(landmark_file)
                        logger.info(f"✅ Loaded dlib landmarks: {landmark_file}")
                        break
                
                if self.landmark_predictor is None:
                    logger.warning("⚠️  Facial landmark predictor not found. Download from:")
                    logger.warning("   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                    
            except Exception as e:
                logger.error(f"❌ dlib initialization failed: {e}")
        
        # Classification thresholds (fine-tuned from working code)
        self.thresholds = {
            'lying_aspect_ratio': 1.2,      # Height/width ratio for lying
            'sitting_aspect_ratio': 1.8,    # Height/width ratio for sitting
            'lying_center_y': 0.7,          # Vertical position threshold
            'head_angle_lying': 45,          # Head angle for lying detection
            'temporal_smoothing': 0.7        # Weight for temporal consistency
        }
    
    def analyze_survivor_pose(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """Comprehensive pose analysis using multiple features (from working code)"""
        x1, y1, x2, y2 = bbox
        person_roi = frame[y1:y2, x1:x2]
        
        # Basic geometric analysis
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = height / width if width > 0 else 0
        center_y = (y1 + y2) / 2
        frame_height = frame.shape[0]
        center_y_ratio = center_y / frame_height if frame_height > 0 else 0.5
        
        # Initialize pose analysis
        pose_analysis = {
            'aspect_ratio': aspect_ratio,
            'center_y_ratio': center_y_ratio,
            'head_angle': 0.0,
            'body_angle': 0.0,
            'facial_landmarks': None,
            'confidence_scores': {
                'standing': 0.0,
                'sitting': 0.0,
                'lying_down': 0.0
            }
        }
        
        # Enhanced analysis with dlib
        if self.face_detector and self.landmark_predictor and person_roi.size > 0:
            try:
                pose_analysis.update(self.analyze_with_dlib(person_roi))
            except Exception as e:
                logger.debug(f"dlib analysis error: {e}")
        
        # Multi-feature classification (from working code)
        action, confidence = self.classify_action_multifeature(pose_analysis)
        pose_analysis['predicted_action'] = action
        pose_analysis['classification_confidence'] = confidence
        
        return pose_analysis
    
    def analyze_with_dlib(self, person_roi: np.ndarray) -> Dict:
        """Advanced pose analysis using dlib facial landmarks"""
        gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the person ROI
        faces = self.face_detector(gray_roi)
        
        analysis = {
            'head_angle': 0.0,
            'body_angle': 0.0,
            'facial_landmarks': None
        }
        
        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda f: f.width() * f.height())
            
            # Get facial landmarks
            landmarks = self.landmark_predictor(gray_roi, face)
            landmark_points = np.array([[p.x, p.y] for p in landmarks.parts()])
            analysis['facial_landmarks'] = landmark_points
            
            # Calculate head orientation
            analysis['head_angle'] = self.calculate_head_angle(landmark_points)
            analysis['body_angle'] = self.estimate_body_angle(landmark_points, person_roi.shape)
        
        return analysis
    
    def calculate_head_angle(self, landmarks: np.ndarray) -> float:
        """Calculate head tilt angle from facial landmarks"""
        try:
            # Use eye corners to determine head orientation
            left_eye_corner = landmarks[36]   # Left eye outer corner
            right_eye_corner = landmarks[45]  # Right eye outer corner
            
            # Calculate angle between eyes
            dx = right_eye_corner[0] - left_eye_corner[0]
            dy = right_eye_corner[1] - left_eye_corner[1]
            
            angle = math.degrees(math.atan2(dy, dx))
            return abs(angle)
            
        except Exception as e:
            logger.debug(f"Head angle calculation error: {e}")
            return 0.0
    
    def estimate_body_angle(self, landmarks: np.ndarray, roi_shape: Tuple) -> float:
        """Estimate body orientation from head position and ROI shape"""
        try:
            # Use nose position relative to ROI
            nose_tip = landmarks[30]  # Nose tip
            roi_height, roi_width = roi_shape[:2]
            
            # Calculate relative position
            nose_x_ratio = nose_tip[0] / roi_width
            nose_y_ratio = nose_tip[1] / roi_height
            
            # Estimate body angle based on head position
            if nose_y_ratio > 0.8:  # Head at bottom of ROI
                return 45  # Likely lying down
            elif nose_y_ratio < 0.3:  # Head at top of ROI
                return 0   # Likely standing
            else:
                return 15  # Likely sitting
                
        except Exception as e:
            logger.debug(f"Body angle estimation error: {e}")
            return 0.0
    
    def classify_action_multifeature(self, pose_analysis: Dict) -> Tuple[str, float]:
        """Multi-feature action classification with confidence scoring (from working code)"""
        aspect_ratio = pose_analysis['aspect_ratio']
        center_y_ratio = pose_analysis['center_y_ratio']
        head_angle = pose_analysis['head_angle']
        body_angle = pose_analysis['body_angle']
        
        # Initialize confidence scores
        scores = {
            'lying_down': 0.0,
            'sitting': 0.0,
            'standing': 0.0
        }
        
        # Lying down detection (multiple indicators)
        lying_indicators = 0
        if aspect_ratio < self.thresholds['lying_aspect_ratio']:
            scores['lying_down'] += 0.4
            lying_indicators += 1
        
        if center_y_ratio > self.thresholds['lying_center_y']:
            scores['lying_down'] += 0.3
            lying_indicators += 1
        
        if head_angle > self.thresholds['head_angle_lying']:
            scores['lying_down'] += 0.3
            lying_indicators += 1
        
        # Sitting detection
        if (self.thresholds['lying_aspect_ratio'] <= aspect_ratio < self.thresholds['sitting_aspect_ratio'] 
            and 0.4 <= center_y_ratio <= 0.8):
            scores['sitting'] += 0.6
            
        if 10 <= head_angle <= 30:  # Moderate head tilt
            scores['sitting'] += 0.2
            
        if 10 <= body_angle <= 25:  # Body slightly angled
            scores['sitting'] += 0.2
        
        # Standing detection
        if aspect_ratio >= self.thresholds['sitting_aspect_ratio']:
            scores['standing'] += 0.5
            
        if center_y_ratio <= 0.6:  # Person in upper part of frame
            scores['standing'] += 0.3
            
        if head_angle <= 15:  # Head relatively straight
            scores['standing'] += 0.2
        
        # Apply temporal smoothing
        if len(self.pose_history) > 0:
            prev_action = self.pose_history[-1]['action']
            for action in scores:
                if action == prev_action:
                    scores[action] += 0.1  # Bonus for consistency
        
        # Determine final classification
        predicted_action = max(scores, key=scores.get)
        confidence = scores[predicted_action]
        
        # Ensure minimum confidence and fallback logic
        if confidence < 0.3:
            # Fallback to basic geometric analysis
            if aspect_ratio < 1.0:
                predicted_action = 'lying_down'
                confidence = 0.4
            elif aspect_ratio < 1.6:
                predicted_action = 'sitting'
                confidence = 0.4
            else:
                predicted_action = 'standing'
                confidence = 0.4
        
        # Store in history for temporal smoothing
        self.pose_history.append({
            'action': predicted_action,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        return predicted_action, min(confidence, 1.0)

class DisasterResponseSystem:
    """
    Complete Disaster Survival Drone Detection System
    Specialized for search, rescue, and survival assessment
    """
    
    def __init__(self, camera_id=0, resolution=(1280, 720), target_fps=30):
        """Initialize the disaster response detection system"""
        self.camera_id = camera_id
        self.resolution = resolution
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # System state
        self.running = False
        self.paused = False
        self.show_performance = True
        self.show_priorities = True
        self.emergency_mode = False  # Highlights critical detections
        self.recording = False
        
        # Detection filtering
        self.enabled_categories = set(DISASTER_CATEGORIES.keys())  # All enabled by default
        self.minimum_confidence = 0.5
        self.priority_threshold = 5  # Only show priority 5+ objects
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        self.last_fps_time = time.time()
        self.frame_count = 0
        
        # Detection statistics
        self.detection_stats = defaultdict(int)
        self.survivor_analyses = {}  # person_id -> SurvivorAnalysis
        
        # AI Models
        self.yolo_model = None
        self.pose_classifier = None
        self.face_cascade = None
        
        # Enhanced pose analyzer (using working code)
        self.pose_classifier = EnhancedSurvivorPoseAnalyzer()
        
        # Tracking
        self.trackers = {}
        self.next_object_id = 0
        
        # Emergency alerts
        self.emergency_alerts = deque(maxlen=50)
        self.critical_detections = deque(maxlen=20)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize models
        self.initialize_models()
        
        # Create output directory
        self.output_dir = Path("disaster_detection_output")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🚁 Disaster Survival Drone Detection System Initialized")
        print(f"📹 Target Resolution: {resolution[0]}x{resolution[1]} @ {target_fps}fps")
        print("🆘 Emergency response priorities active")
        print("🎮 Controls: Q=Quit, SPACE=Pause, E=Emergency Mode, P=Priorities, F=Filter")
    
    def initialize_models(self):
        """Initialize AI models for disaster detection"""
        logger.info("🔄 Loading disaster detection models...")
        
        # Initialize YOLO for multi-object detection
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                self.yolo_model.fuse()
                logger.info("✅ YOLO model loaded for disaster object detection")
            except Exception as e:
                logger.warning(f"⚠️  YOLO loading failed: {e}")
                self.yolo_model = None
        
        # Initialize face detection for emotion analysis
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("✅ Face detection ready for distress analysis")
        except Exception as e:
            logger.warning(f"⚠️  Face detection failed: {e}")
    
    def setup_camera(self):
        """Setup camera optimized for drone operations"""
        logger.info(f"🎥 Initializing drone camera {self.camera_id}")
        
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            logger.error(f"❌ Cannot open camera {self.camera_id}")
            return None
        
        # Optimize for aerial/drone footage
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal latency
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Drone-specific optimizations
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Stable exposure
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for stability
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"📹 Drone camera ready: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
        return cap
    
    def detect_disaster_objects(self, frame) -> List[DisasterDetection]:
        """Detect all disaster-relevant objects in frame"""
        detections = []
        
        if self.yolo_model:
            try:
                # Run YOLO detection
                results = self.yolo_model(frame, verbose=False, conf=self.minimum_confidence)
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # Only process disaster-relevant objects
                            if class_id in DISASTER_OBJECTS:
                                class_name = DISASTER_OBJECTS[class_id]
                                category = self.get_object_category(class_id)
                                priority = EMERGENCY_PRIORITIES.get(category, 1)
                                
                                # Skip if category is disabled or priority too low
                                if (category not in self.enabled_categories or 
                                    priority < self.priority_threshold):
                                    continue
                                
                                detection = DisasterDetection(
                                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                                    confidence=confidence,
                                    class_id=class_id,
                                    class_name=class_name,
                                    category=category,
                                    priority=priority,
                                    object_id=i,
                                    timestamp=time.time()
                                )
                                
                                # Special analysis for humans (survivors)
                                if class_id == 0:  # person
                                    self.analyze_survivor(frame, detection)
                                
                                # Resource condition analysis
                                elif category in ['survival_resources', 'equipment']:
                                    self.analyze_resource_condition(frame, detection)
                                
                                detections.append(detection)
                                self.detection_stats[category] += 1
                                
            except Exception as e:
                logger.debug(f"YOLO detection error: {e}")
        else:
            # Simulated disaster detection for testing
            detections = self.simulate_disaster_detection(frame)
        
        return detections
    
    def get_object_category(self, class_id: int) -> str:
        """Get disaster category for object class"""
        for category, class_ids in DISASTER_CATEGORIES.items():
            if class_id in class_ids:
                return category
        return 'unknown'
    
    def analyze_survivor(self, frame: np.ndarray, detection: DisasterDetection):
        """Analyze survivor condition and distress level using enhanced pose analysis"""
        if self.pose_classifier:
            try:
                # Use the enhanced pose analysis (from working system)
                pose_analysis = self.pose_classifier.analyze_survivor_pose(frame, detection.bbox)
                
                # Extract all the enhanced pose data
                detection.action = pose_analysis.get('predicted_action', 'unknown')
                detection.head_angle = pose_analysis.get('head_angle', 0.0)
                detection.body_angle = pose_analysis.get('body_angle', 0.0)
                detection.aspect_ratio = pose_analysis.get('aspect_ratio', 0.0)
                detection.center_y_ratio = pose_analysis.get('center_y_ratio', 0.0)
                detection.classification_confidence = pose_analysis.get('classification_confidence', 0.0)
                
                # Determine distress level using multiple indicators
                distress_indicators = []
                distress_score = 0
                
                # Primary distress indicator: lying down
                if detection.action == 'lying_down':
                    distress_indicators.append('lying_down')
                    distress_score += 0.7
                    detection.distress_level = 'high'
                
                # Secondary indicators from enhanced pose analysis
                if detection.head_angle > 45:
                    distress_indicators.append('head_severely_tilted')
                    distress_score += 0.4
                elif detection.head_angle > 30:
                    distress_indicators.append('head_tilted')
                    distress_score += 0.2
                
                # Body position indicators
                if detection.center_y_ratio > 0.7:  # Low in frame
                    distress_indicators.append('low_position')
                    distress_score += 0.3
                
                if detection.aspect_ratio < 1.0:  # Very horizontal
                    distress_indicators.append('horizontal_posture')
                    distress_score += 0.4
                
                # Low classification confidence may indicate unusual pose
                if detection.classification_confidence < 0.5:
                    distress_indicators.append('unusual_posture')
                    distress_score += 0.2
                
                # Check for no movement (temporal tracking)
                if detection.object_id in self.survivor_analyses:
                    prev_analysis = self.survivor_analyses[detection.object_id]
                    time_since_movement = time.time() - prev_analysis.last_movement
                    if time_since_movement > 30:  # No movement for 30 seconds
                        distress_indicators.append('no_movement')
                        distress_score += 0.5
                
                # Determine final distress level based on combined score
                if distress_score >= 0.8:
                    detection.distress_level = 'critical'
                    detection.needs_immediate_attention = True
                elif distress_score >= 0.5:
                    detection.distress_level = 'high' 
                    detection.needs_immediate_attention = True
                elif distress_score >= 0.3:
                    detection.distress_level = 'medium'
                else:
                    detection.distress_level = 'low'
                
                # Update survivor analysis tracking
                survivor_analysis = SurvivorAnalysis(
                    person_id=detection.object_id,
                    distress_indicators=distress_indicators,
                    mobility_status='immobile' if 'lying_down' in distress_indicators else 'mobile',
                    estimated_condition='critical' if distress_score >= 0.8 else 'injured' if distress_score >= 0.5 else 'good',
                    last_movement=time.time(),
                    requires_evacuation=detection.needs_immediate_attention,
                    medical_priority=min(5, int(distress_score * 5) + 1)
                )
                self.survivor_analyses[detection.object_id] = survivor_analysis
                
                # Log emergency alerts for immediate attention cases
                if detection.needs_immediate_attention:
                    alert = f"SURVIVOR DISTRESS: {', '.join(distress_indicators)} - Priority {survivor_analysis.medical_priority} at {time.strftime('%H:%M:%S')}"
                    self.emergency_alerts.append(alert)
                    self.critical_detections.append(detection)
                    logger.warning(f"🆘 {alert}")
                
            except Exception as e:
                logger.debug(f"Survivor analysis error: {e}")
                # Fallback to basic analysis
                detection.action = 'unknown'
                detection.distress_level = 'unknown'
    
    def analyze_resource_condition(self, frame: np.ndarray, detection: DisasterDetection):
        """Analyze condition and accessibility of survival resources"""
        try:
            # Basic condition analysis based on detection confidence and context
            if detection.confidence > 0.8:
                detection.resource_condition = 'good'
                detection.accessibility = 'accessible'
            elif detection.confidence > 0.6:
                detection.resource_condition = 'fair'
                detection.accessibility = 'accessible'
            else:
                detection.resource_condition = 'poor'
                detection.accessibility = 'blocked'
                
        except Exception as e:
            logger.debug(f"Resource analysis error: {e}")
    
    def simulate_disaster_detection(self, frame) -> List[DisasterDetection]:
        """Simulate disaster object detection for testing with enhanced pose simulation"""
        detections = []
        h, w = frame.shape[:2]
        
        # Simulate realistic disaster scenario
        disaster_objects = [
            (0, 'person', 'survivors', 10),      # Person - highest priority
            (16, 'dog', 'animals', 7),           # Dog - pet rescue
            (2, 'car', 'vehicles', 6),           # Car - obstacle/transport
            (39, 'bottle', 'survival_resources', 8), # Water bottle
            (67, 'cell phone', 'communication', 9),  # Phone
        ]
        
        num_objects = np.random.randint(1, 4)  # 1-3 objects per frame
        
        for _ in range(num_objects):
            if disaster_objects:
                class_id, name, category, priority = np.random.choice(disaster_objects)
                
                # Enhanced positioning based on object type (from working code)
                if category == 'survivors':
                    # More realistic pose-based positioning
                    action = np.random.choice(['standing', 'sitting', 'lying_down'], p=[0.5, 0.4, 0.1])
                    
                    if action == 'standing':
                        # Standing: taller, more vertical
                        x1 = np.random.randint(w//4, 3*w//4)
                        y1 = np.random.randint(50, h - 300)
                        width = np.random.randint(70, 100)
                        height = np.random.randint(200, 280)
                    elif action == 'sitting':
                        # Sitting: shorter, more in middle
                        x1 = np.random.randint(w//4, 3*w//4)
                        y1 = np.random.randint(h//3, h - 180)
                        width = np.random.randint(80, 120)
                        height = np.random.randint(120, 180)
                    else:  # lying_down
                        # Lying: wider, lower in frame
                        x1 = np.random.randint(w//4, 3*w//4)
                        y1 = np.random.randint(h//2, h - 120)
                        width = np.random.randint(150, 250)
                        height = np.random.randint(80, 120)
                else:
                    # Other objects random positioning
                    x1 = np.random.randint(50, w - 150)
                    y1 = np.random.randint(50, h - 100)
                    width, height = np.random.randint(40, 120), np.random.randint(40, 120)
                    action = None
                
                x2 = min(x1 + width, w - 1)
                y2 = min(y1 + height, h - 1)
                
                detection = DisasterDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=np.random.uniform(0.6, 0.95),
                    class_id=class_id,
                    class_name=name,
                    category=category,
                    priority=priority,
                    object_id=len(detections),
                    timestamp=time.time()
                )
                
                # Add enhanced pose simulation for people
                if class_id == 0 and action:
                    detection.action = action
                    detection.aspect_ratio = height / width if width > 0 else 0
                    detection.center_y_ratio = (y1 + height/2) / h
                    detection.classification_confidence = np.random.uniform(0.8, 0.95)
                    
                    if action == 'lying_down':
                        detection.distress_level = 'high'
                        detection.needs_immediate_attention = True
                        detection.head_angle = np.random.uniform(30, 60)
                    elif action == 'sitting':
                        detection.distress_level = 'medium'
                        detection.head_angle = np.random.uniform(5, 20)
                    else:  # standing
                        detection.distress_level = 'low'
                        detection.head_angle = np.random.uniform(0, 10)
                
                detections.append(detection)
        
        return detections
    
    def update_tracking(self, detections: List[DisasterDetection]) -> List[DisasterDetection]:
        """Update object tracking with disaster-specific logic"""
        tracked_detections = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Find best matching tracker (priority for survivors)
            best_match = None
            min_distance = float('inf')
            
            for track_id, track_data in self.trackers.items():
                if track_data['class_id'] != detection.class_id:
                    continue  # Only match same object types
                
                track_center = track_data['center']
                distance = np.sqrt(
                    (center[0] - track_center[0])**2 + 
                    (center[1] - track_center[1])**2
                )
                
                # Bonus for high priority objects (survivors)
                if detection.priority >= 8:
                    distance *= 0.7  # Make survivors easier to track
                
                if distance < min_distance and distance < 150:
                    min_distance = distance
                    best_match = track_id
            
            if best_match is not None:
                # Update existing tracker
                self.trackers[best_match]['center'] = center
                self.trackers[best_match]['last_seen'] = time.time()
                detection.object_id = best_match
                detection.tracking_confidence = max(0.1, 1.0 - min_distance / 150)
            else:
                # Create new tracker
                self.trackers[self.next_object_id] = {
                    'center': center,
                    'last_seen': time.time(),
                    'first_seen': time.time(),
                    'class_id': detection.class_id,
                    'category': detection.category
                }
                detection.object_id = self.next_object_id
                detection.tracking_confidence = 1.0
                self.next_object_id += 1
            
            tracked_detections.append(detection)
        
        # Remove old trackers (shorter timeout for critical objects)
        current_time = time.time()
        timeout = 1.5  # Shorter timeout for real-time response
        old_trackers = [
            tid for tid, data in self.trackers.items() 
            if current_time - data['last_seen'] > timeout
        ]
        for tid in old_trackers:
            del self.trackers[tid]
        
        return tracked_detections
    
    def draw_disaster_detections(self, frame, detections: List[DisasterDetection]) -> np.ndarray:
        """Draw disaster detections with emergency priorities"""
        display_frame = frame.copy()
        
        # Sort by priority (highest first)
        detections.sort(key=lambda d: d.priority, reverse=True)
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Color based on category and priority
            base_color = PRIORITY_COLORS.get(detection.category, (128, 128, 128))
            
            # Intensity based on priority and confidence
            intensity = min(1.0, (detection.priority / 10) * detection.confidence)
            color = tuple(int(c * intensity) for c in base_color)
            
            # Thickness based on priority
            thickness = max(2, min(6, detection.priority))
            
            # Emergency mode: blinking for critical detections
            if (self.emergency_mode and detection.needs_immediate_attention and 
                int(time.time() * 3) % 2):  # Blink effect
                color = (0, 0, 255)  # Red for emergency
                thickness = 6
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw priority indicator
            self.draw_priority_indicator(display_frame, detection)
            
            # Draw detailed label
            self.draw_disaster_label(display_frame, detection)
            
            # Draw survivor analysis
            if detection.category == 'survivors':
                self.draw_survivor_analysis(display_frame, detection)
        
        # Draw emergency information
        self.draw_emergency_panel(display_frame, detections)
        
        # Draw performance and statistics
        if self.show_performance:
            self.draw_performance_stats(display_frame)
        
        # Draw control information
        self.draw_controls(display_frame)
        
        return display_frame
    
    def draw_priority_indicator(self, frame, detection: DisasterDetection):
        """Draw priority level indicator"""
        x1, y1, x2, y2 = detection.bbox
        
        # Priority badge
        priority_text = f"P{detection.priority}"
        priority_color = (0, 0, 255) if detection.priority >= 8 else (255, 255, 0)
        
        # Priority badge background
        cv2.circle(frame, (x2 - 15, y1 + 15), 12, priority_color, -1)
        cv2.circle(frame, (x2 - 15, y1 + 15), 12, (0, 0, 0), 2)
        
        # Priority text
        cv2.putText(frame, str(detection.priority), (x2 - 20, y1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def draw_disaster_label(self, frame, detection: DisasterDetection):
        """Draw comprehensive disaster object label with enhanced survivor info"""
        x1, y1, x2, y2 = detection.bbox
        
        # Main label
        main_text = f"{detection.class_name.upper()}"
        category_text = f"[{detection.category}]"
        confidence_text = f"{detection.confidence:.2f}"
        
        # Enhanced status information for survivors (same as second code)
        if detection.category == 'survivors':
            if detection.action and detection.action != 'unknown':
                status_text = f"Action: {detection.action.upper()} ({detection.classification_confidence:.2f})"
            else:
                status_text = "Status: Analyzing..."
        else:
            status_text = f"Condition: {detection.resource_condition}"
        
        # Calculate label dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        labels = [main_text, category_text, confidence_text, status_text]
        label_heights = []
        max_width = 0
        
        for label in labels:
            (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_heights.append(h)
            max_width = max(max_width, w)
        
        # Background for labels
        total_height = sum(label_heights) + len(labels) * 5
        cv2.rectangle(frame, (x1, y1 - total_height - 10), 
                     (x1 + max_width + 15, y1), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1 - total_height - 10), 
                     (x1 + max_width + 15, y1), (255, 255, 255), 1)
        
        # Draw labels with action-specific colors for survivors
        y_offset = y1 - total_height
        colors = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        
        # Special color for action text based on action type
        if detection.category == 'survivors' and detection.action:
            action_text_colors = {
                'standing': (0, 255, 0),    # Green
                'sitting': (255, 255, 0),   # Yellow
                'lying_down': (255, 0, 0),  # Red
                'unknown': (255, 255, 255)  # White
            }
            colors[3] = action_text_colors.get(detection.action, (255, 255, 255))
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            cv2.putText(frame, label, (x1 + 5, y_offset + label_heights[i]),
                       font, font_scale * (0.8 if i > 0 else 1.0), color, thickness)
            y_offset += label_heights[i] + 5
    
    def draw_survivor_analysis(self, frame, detection: DisasterDetection):
        """Draw survivor-specific analysis visualization"""
        if detection.needs_immediate_attention:
            x1, y1, x2, y2 = detection.bbox
            
            # Emergency indicator
            cv2.putText(frame, "🆘 NEEDS HELP", (x1, y2 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            
            # Distress level bar
            bar_width = x2 - x1
            bar_height = 8
            bar_y = y2 + 40
            
            # Background bar
            cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_height), (50, 50, 50), -1)
            
            # Distress level fill
            distress_levels = {'low': 0.3, 'medium': 0.6, 'high': 0.8, 'critical': 1.0}
            fill_ratio = distress_levels.get(detection.distress_level, 0.5)
            fill_width = int(bar_width * fill_ratio)
            fill_color = (0, 255, 255) if fill_ratio < 0.7 else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, bar_y), (x1 + fill_width, bar_y + bar_height), 
                         fill_color, -1)
    
    def draw_emergency_panel(self, frame, detections: List[DisasterDetection]):
        """Draw emergency information panel"""
        h, w = frame.shape[:2]
        panel_x = w - 300
        panel_y = 10
        
        # Emergency summary
        critical_count = sum(1 for d in detections if d.needs_immediate_attention)
        survivor_count = sum(1 for d in detections if d.category == 'survivors')
        resource_count = sum(1 for d in detections if d.category == 'survival_resources')
        
        emergency_info = [
            f"🆘 Critical: {critical_count}",
            f"👥 Survivors: {survivor_count}",
            f"💧 Resources: {resource_count}",
            f"📊 Total Objects: {len(detections)}",
        ]
        
        # Draw panel background
        panel_height = len(emergency_info) * 25 + 20
        cv2.rectangle(frame, (panel_x, panel_y), (w - 10, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (w - 10, panel_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Draw emergency info
        for i, info in enumerate(emergency_info):
            color = (0, 0, 255) if i == 0 and critical_count > 0 else (255, 255, 255)
            cv2.putText(frame, info, (panel_x + 10, panel_y + 20 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Recent alerts
        if self.emergency_alerts:
            alert_y = panel_y + panel_height + 20
            cv2.putText(frame, "Recent Alerts:", (panel_x, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for i, alert in enumerate(list(self.emergency_alerts)[-3:]):  # Last 3 alerts
                cv2.putText(frame, alert[:40] + "...", (panel_x, alert_y + 25 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_performance_stats(self, frame):
        """Draw system performance statistics"""
        stats_info = [
            f"FPS: {np.mean(self.fps_counter):.1f}" if self.fps_counter else "FPS: 0",
            f"Processing: {np.mean(self.processing_times):.1f}ms" if self.processing_times else "Processing: 0ms",
            f"Tracking: {len(self.trackers)} objects",
            f"Emergency Mode: {'ON' if self.emergency_mode else 'OFF'}",
        ]
        
        for i, stat in enumerate(stats_info):
            cv2.putText(frame, stat, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_controls(self, frame):
        """Draw control instructions for disaster operations"""
        h, w = frame.shape[:2]
        controls_y = h - 80
        
        controls = [
            "🚁 DISASTER DRONE CONTROLS:",
            "Q=Quit | SPACE=Pause | E=Emergency Mode | P=Priorities | F=Filter Categories",
            "S=Screenshot | R=Record | C=Clear Alerts | H=Help"
        ]
        
        for i, control in enumerate(controls):
            color = (255, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(frame, control, (10, controls_y + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def handle_keyboard_input(self, key):
        """Handle keyboard input for disaster operations"""
        if key == ord('q') or key == 27:  # Q or ESC
            self.running = False
            logger.info("🛑 Disaster detection system shutdown")
            
        elif key == ord(' '):  # SPACE
            self.paused = not self.paused
            status = "paused" if self.paused else "resumed"
            logger.info(f"⏸️  System {status}")
            
        elif key == ord('e'):  # E - Emergency mode
            self.emergency_mode = not self.emergency_mode
            status = "activated" if self.emergency_mode else "deactivated"
            logger.info(f"🆘 Emergency mode {status}")
            
        elif key == ord('p'):  # P - Show priorities
            self.show_priorities = not self.show_priorities
            logger.info(f"📊 Priority display: {self.show_priorities}")
            
        elif key == ord('f'):  # F - Filter categories
            self.cycle_category_filters()
            
        elif key == ord('s'):  # S - Screenshot
            self.save_screenshot()
            
        elif key == ord('r'):  # R - Record
            self.toggle_recording()
            
        elif key == ord('c'):  # C - Clear alerts
            self.emergency_alerts.clear()
            self.critical_detections.clear()
            logger.info("🧹 Emergency alerts cleared")
            
        elif key == ord('h'):  # H - Help
            self.show_help()
    
    def cycle_category_filters(self):
        """Cycle through different category filter combinations"""
        filter_presets = [
            set(DISASTER_CATEGORIES.keys()),  # All categories
            {'survivors', 'communication'},    # Critical only
            {'survivors', 'animals'},         # Life forms only
            {'survivors', 'survival_resources', 'communication'},  # Essential
        ]
        
        current_index = 0
        for i, preset in enumerate(filter_presets):
            if preset == self.enabled_categories:
                current_index = i
                break
        
        next_index = (current_index + 1) % len(filter_presets)
        self.enabled_categories = filter_presets[next_index]
        
        enabled_list = ', '.join(self.enabled_categories)
        logger.info(f"🔍 Filter changed to: {enabled_list}")
    
    def save_screenshot(self):
        """Save screenshot with disaster context"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"disaster_detection_{timestamp}.jpg"
        
        if hasattr(self, 'current_frame'):
            cv2.imwrite(str(filename), self.current_frame)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'detection_stats': dict(self.detection_stats),
                'emergency_alerts': list(self.emergency_alerts)[-5:],  # Last 5 alerts
                'critical_detections_count': len(self.critical_detections)
            }
            
            metadata_file = self.output_dir / f"disaster_detection_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"📸 Disaster screenshot saved: {filename}")
        else:
            logger.warning("⚠️  No frame available for screenshot")
    
    def toggle_recording(self):
        """Toggle video recording"""
        self.recording = not self.recording
        
        if self.recording:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.video_filename = self.output_dir / f"disaster_recording_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(self.video_filename),
                fourcc, 
                self.target_fps,
                self.resolution
            )
            logger.info(f"🎥 Recording started: {self.video_filename}")
        else:
            if hasattr(self, 'video_writer'):
                self.video_writer.release()
                logger.info(f"⏹️  Recording stopped: {self.video_filename}")
    
    def show_help(self):
        """Display help information in console"""
        help_text = """
🚁 DISASTER SURVIVAL DRONE DETECTION SYSTEM - HELP
================================================================

OBJECT CATEGORIES:
• Survivors (Priority 10): People needing rescue
• Communication (Priority 9): Phones, signaling devices
• Survival Resources (Priority 8): Water, food, supplies
• Animals (Priority 7): Pets, livestock rescue
• Vehicles (Priority 6): Transportation, obstacles
• Equipment (Priority 5): Tools, useful items
• Shelter (Priority 4): Shelter materials, furniture

EMERGENCY INDICATORS:
• Red blinking boxes: Critical distress detected
• Priority numbers: 1-10 scale (10 = highest)
• Status messages: Survivor condition analysis

CONTROLS:
• E: Toggle Emergency Mode (highlights critical cases)
• F: Cycle through category filters
• C: Clear emergency alerts
• S: Save screenshot with metadata
• R: Record video for evidence

DISASTER RESPONSE WORKFLOW:
1. Survey area with drone
2. Monitor for survivors (green boxes, priority 10)
3. Check distress indicators (lying down, no movement)
4. Locate survival resources (water, food, supplies)
5. Document findings with screenshots/recording
================================================================
        """
        print(help_text)
        logger.info("📖 Help information displayed in console")
    
    def run(self):
        """Main execution loop for disaster detection"""
        logger.info("🚁 Starting Disaster Survival Drone Detection System")
        logger.info("🆘 Emergency response mode active")
        logger.info("🎮 Press 'H' for help, 'Q' to quit")
        
        # Setup camera
        cap = self.setup_camera()
        if cap is None:
            logger.error("❌ Drone camera setup failed")
            return False
        
        # Create window
        cv2.namedWindow('Disaster Survival Drone Detection', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('Disaster Survival Drone Detection', 100, 100)
        
        self.running = True
        
        try:
            while self.running:
                start_time = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("⚠️  Failed to capture frame")
                    continue
                
                # Skip processing if paused
                if self.paused:
                    cv2.imshow('Disaster Survival Drone Detection', frame)
                    key = cv2.waitKey(1) & 0xFF
                    self.handle_keyboard_input(key)
                    continue
                
                # Detect disaster objects
                detections = self.detect_disaster_objects(frame)
                detections = self.update_tracking(detections)
                
                # Draw results
                display_frame = self.draw_disaster_detections(frame, detections)
                self.current_frame = display_frame
                
                # Record if enabled
                if self.recording and hasattr(self, 'video_writer'):
                    self.video_writer.write(display_frame)
                
                # Update performance stats
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time * 1000)
                if len(self.processing_times) > 30:
                    self.processing_times.popleft()
                
                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    fps = self.frame_count / (current_time - self.last_fps_time)
                    self.fps_counter.append(fps)
                    if len(self.fps_counter) > 30:
                        self.fps_counter.popleft()
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # Display frame
                cv2.imshow('Disaster Survival Drone Detection', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                self.handle_keyboard_input(key)
                
                # Frame rate control
                elapsed = time.time() - start_time
                sleep_time = max(0, self.frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ System error: {e}")
        finally:
            self.cleanup(cap)
        
        return True
    
    def cleanup(self, cap):
        """Clean up disaster detection system"""
        logger.info("🧹 Cleaning up disaster detection system...")
        
        self.running = False
        
        # Stop recording if active
        if self.recording and hasattr(self, 'video_writer'):
            self.video_writer.release()
        
        # Release camera
        cap.release()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=True)
        
        # Print final disaster report
        logger.info("📊 FINAL DISASTER DETECTION REPORT:")
        logger.info(f"   Total Detection Sessions: {sum(self.detection_stats.values())}")
        for category, count in self.detection_stats.items():
            logger.info(f"   {category.title()}: {count} detected")
        logger.info(f"   Emergency Alerts Generated: {len(self.emergency_alerts)}")
        logger.info(f"   Critical Detections: {len(self.critical_detections)}")
        
        if self.critical_detections:
            logger.warning("⚠️  CRITICAL DETECTIONS SUMMARY:")
            for detection in self.critical_detections:
                logger.warning(f"     • {detection.class_name} - {detection.distress_level} distress")

def main():
    """Main entry point for disaster survival drone detection"""
    print("🚁 DISASTER SURVIVAL DRONE DETECTION SYSTEM")
    print("=" * 70)
    print("🆘 Specialized for Search, Rescue & Survival Assessment")
    print("🎯 Detecting: Survivors, Animals, Resources, Communication")
    print("⚡ Emergency Priority System | Real-time Distress Analysis")
    print("=" * 70)
    
    # Configuration
    config = {
        'camera_id': 0,
        'resolution': (1280, 720),
        'target_fps': 30
    }
    
    # Command line arguments
    import sys
    if len(sys.argv) > 1:
        try:
            config['camera_id'] = int(sys.argv[1])
            print(f"📹 Using camera ID: {config['camera_id']}")
        except ValueError:
            print("⚠️  Invalid camera ID, using default (0)")
    
    try:
        # Create and run disaster detection system
        detector = DisasterResponseSystem(**config)
        success = detector.run()
        
        if success:
            print("✅ Disaster detection system completed successfully")
            return 0
        else:
            print("❌ System failed to initialize")
            return 1
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())