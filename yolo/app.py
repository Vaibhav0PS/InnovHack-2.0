import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
import threading
import time
import queue
from collections import defaultdict, deque
import os
import logging
from ultralytics import YOLO
import torch
from datetime import datetime, timedelta
import json

# Initialize Flask app
app = Flask(__name__)

# Configuration - Enhanced for YOLOv11
DETECTION_INTERVAL = 5  # Process every 5th frame (YOLOv11 is faster)
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
FRAME_SKIP = 1  # Reduced frame skipping due to better performance
MAX_QUEUE_SIZE = 10
TRACKING_ENABLED = True
POSE_ESTIMATION = False  # Set to True for pose detection
SEGMENTATION = False  # Set to True for instance segmentation

# Advanced analytics configuration
ANALYTICS_WINDOW = 300  # 5 minutes of data retention
ALERT_THRESHOLD = {'person': 5, 'car': 3}  # Alert thresholds for different objects
ZONE_DETECTION = True  # Enable zone-based detection

# Use in-memory storage with time-series data
analytics_data = defaultdict(list)  # Store time-series data
current_detections = defaultdict(int)
object_trajectories = defaultdict(lambda: deque(maxlen=50))  # Store object paths
alert_history = deque(maxlen=100)
analytics_lock = threading.Lock()

# Detection zones (can be configured via web interface)
detection_zones = [
    {"name": "Entry Zone", "polygon": [(50, 50), (200, 50), (200, 200), (50, 200)], "active": True},
    {"name": "Restricted Area", "polygon": [(300, 100), (500, 100), (500, 300), (300, 300)], "active": True}
]

# Load YOLOv11 models
try:
    # Load different YOLOv11 models based on requirements
    if POSE_ESTIMATION:
        model = YOLO('yolo11n-pose.pt')  # Pose estimation model
        print("✓ YOLOv11 Pose model loaded")
    elif SEGMENTATION:
        model = YOLO('yolo11n-seg.pt')  # Segmentation model
        print("✓ YOLOv11 Segmentation model loaded")
    else:
        model = YOLO('yolo11n.pt')  # Standard detection model
        print("✓ YOLOv11 Detection model loaded")
    
    # Optimize model for inference
    model.fuse()  # Fuse layers for faster inference
    
    # Enable GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"✓ Using device: {device}")
    
except Exception as e:
    print(f"✗ Error loading YOLOv11 model: {e}")
    print("Please install ultralytics: pip install ultralytics")
    exit(1)

# Thread-safe data structures
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
processed_frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
analytics_queue = queue.Queue(maxsize=100)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ObjectTracker:
    """Enhanced object tracking using YOLOv11's built-in tracking"""
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0
        
    def update_tracks(self, detections):
        """Update object tracks with new detections"""
        current_time = time.time()
        tracked_results = []
        
        for det in detections:
            # Use YOLOv11's tracking capabilities
            if hasattr(det, 'id') and det.id is not None:
                track_id = int(det.id)
                bbox = det.boxes.xyxy[0].cpu().numpy()
                conf = float(det.boxes.conf[0])
                cls = int(det.boxes.cls[0])
                
                tracked_results.append({
                    'id': track_id,
                    'class': model.names[cls],
                    'confidence': conf,
                    'bbox': bbox,
                    'timestamp': current_time
                })
                
                # Store trajectory
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                object_trajectories[track_id].append((center_x, center_y, current_time))
        
        return tracked_results

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def detect_objects_yolov11(frame):
    """Advanced object detection using YOLOv11"""
    try:
        # Run YOLOv11 inference with tracking
        if TRACKING_ENABLED:
            results = model.track(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, 
                                persist=True, tracker="bytetrack.yaml")
        else:
            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        
        detected_objects = []
        current_time = time.time()
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # Get tracking IDs if available
                track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)
                
                for i, (box, conf, cls, track_id) in enumerate(zip(boxes, confidences, classes, track_ids)):
                    x1, y1, x2, y2 = box
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # Zone detection
                    zones_detected = []
                    if ZONE_DETECTION:
                        for zone in detection_zones:
                            if zone['active'] and point_in_polygon((center_x, center_y), zone['polygon']):
                                zones_detected.append(zone['name'])
                    
                    detection_data = {
                        'label': model.names[int(cls)],
                        'confidence': float(conf),
                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        'center': (center_x, center_y),
                        'track_id': int(track_id) if track_id is not None else None,
                        'zones': zones_detected,
                        'timestamp': current_time
                    }
                    
                    # Add pose keypoints if pose estimation is enabled
                    if POSE_ESTIMATION and hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints = result.keypoints.xy[i].cpu().numpy()
                        detection_data['keypoints'] = keypoints.tolist()
                    
                    # Add segmentation mask if segmentation is enabled
                    if SEGMENTATION and hasattr(result, 'masks') and result.masks is not None:
                        mask = result.masks.xy[i]
                        detection_data['mask'] = mask
                    
                    detected_objects.append(detection_data)
        
        return detected_objects
        
    except Exception as e:
        logging.error(f"YOLOv11 detection error: {e}")
        return []

def generate_alerts(detections):
    """Generate alerts based on detection patterns"""
    alerts = []
    current_time = datetime.now()
    
    # Count current detections by class
    detection_counts = defaultdict(int)
    for det in detections:
        detection_counts[det['label']] += 1
    
    # Check threshold alerts
    for obj_class, count in detection_counts.items():
        if obj_class in ALERT_THRESHOLD and count >= ALERT_THRESHOLD[obj_class]:
            alert = {
                'type': 'threshold_exceeded',
                'message': f"High number of {obj_class} detected: {count}",
                'severity': 'warning',
                'timestamp': current_time.isoformat(),
                'data': {'class': obj_class, 'count': count}
            }
            alerts.append(alert)
    
    # Zone intrusion alerts
    for det in detections:
        for zone in det['zones']:
            if 'Restricted' in zone:
                alert = {
                    'type': 'zone_intrusion',
                    'message': f"{det['label']} detected in {zone}",
                    'severity': 'high',
                    'timestamp': current_time.isoformat(),
                    'data': {'class': det['label'], 'zone': zone, 'track_id': det['track_id']}
                }
                alerts.append(alert)
    
    return alerts

def video_capture_thread():
    """Enhanced video capture thread with adaptive frame rate"""
    print("Starting webcam with YOLOv11 optimizations...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Optimize camera settings for YOLOv11
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    frame_count = 0
    fps_counter = deque(maxlen=30)
    last_fps_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            time.sleep(0.01)
            continue
            
        frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        fps_counter.append(current_time)
        if len(fps_counter) > 1:
            fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
        else:
            fps = 0
        
        # Adaptive frame skipping based on processing speed
        if frame_count % DETECTION_INTERVAL != 0:
            continue
        
        # Add frame info for debugging
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"YOLOv11", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add to processing queue
        if not frame_queue.full():
            try:
                frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                # Remove oldest frame if queue is full
                try:
                    frame_queue.get_nowait()
                    frame_queue.put(frame, timeout=0.1)
                except (queue.Empty, queue.Full):
                    pass
        
        time.sleep(0.001)  # Small sleep to prevent CPU overload
    
    cap.release()

def analytics_thread():
    """Enhanced analytics processing thread with YOLOv11"""
    global analytics_data, current_detections
    
    while True:
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        start_time = time.time()
        
        # Perform YOLOv11 detection
        detections = detect_objects_yolov11(frame)
        
        # Generate alerts
        alerts = generate_alerts(detections)
        
        # Update analytics data
        current_time = time.time()
        with analytics_lock:
            # Clear old data
            cutoff_time = current_time - ANALYTICS_WINDOW
            for key in list(analytics_data.keys()):
                analytics_data[key] = [(t, v) for t, v in analytics_data[key] if t > cutoff_time]
            
            # Update current detections
            current_detections.clear()
            for det in detections:
                label = det['label']
                current_detections[label] += 1
                analytics_data[label].append((current_time, 1))
            
            # Store alerts
            for alert in alerts:
                alert_history.append(alert)
        
        # Draw enhanced visualizations
        annotated_frame = frame.copy()
        
        # Draw detection zones
        if ZONE_DETECTION:
            for zone in detection_zones:
                if zone['active']:
                    pts = np.array(zone['polygon'], np.int32)
                    cv2.polylines(annotated_frame, [pts], True, (255, 0, 0), 2)
                    cv2.putText(annotated_frame, zone['name'], 
                              tuple(zone['polygon'][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw detections with enhanced info
        for det in detections:
            x, y, w, h = det['bbox']
            label = det['label']
            conf = det['confidence']
            track_id = det['track_id']
            
            # Color coding based on zones
            color = (0, 255, 0)  # Green by default
            if 'Restricted Area' in det['zones']:
                color = (0, 0, 255)  # Red for restricted areas
            elif det['zones']:
                color = (0, 255, 255)  # Yellow for other zones
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with tracking ID
            label_text = f"{label} {conf:.2f}"
            if track_id is not None:
                label_text += f" ID:{track_id}"
            
            cv2.putText(annotated_frame, label_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory if available
            if track_id and track_id in object_trajectories:
                trajectory = list(object_trajectories[track_id])
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        cv2.line(annotated_frame, 
                               (int(trajectory[i-1][0]), int(trajectory[i-1][1])),
                               (int(trajectory[i][0]), int(trajectory[i][1])), 
                               color, 2)
            
            # Draw pose keypoints if available
            if POSE_ESTIMATION and 'keypoints' in det:
                keypoints = det['keypoints']
                for kp in keypoints:
                    if len(kp) >= 2:
                        cv2.circle(annotated_frame, (int(kp[0]), int(kp[1])), 3, (255, 255, 0), -1)
        
        # Performance info
        processing_time = time.time() - start_time
        cv2.putText(annotated_frame, f"Processing: {processing_time*1000:.1f}ms", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Detections: {len(detections)}", 
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add to output queue
        if not processed_frame_queue.full():
            try:
                processed_frame_queue.put(annotated_frame, timeout=0.1)
            except queue.Full:
                try:
                    processed_frame_queue.get_nowait()
                    processed_frame_queue.put(annotated_frame, timeout=0.1)
                except (queue.Empty, queue.Full):
                    pass
        
        frame_queue.task_done()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                if not processed_frame_queue.empty():
                    frame = processed_frame_queue.get(timeout=0.1)
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    time.sleep(0.01)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Video feed error: {e}")
                time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analytics')
def get_analytics():
    with analytics_lock:
        current_time = time.time()
        
        # Prepare time-series data for charts
        time_series_data = {}
        for label, data_points in analytics_data.items():
            if data_points:
                # Aggregate data into 30-second buckets
                buckets = defaultdict(int)
                for timestamp, value in data_points:
                    bucket = int(timestamp // 30) * 30
                    buckets[bucket] += value
                
                time_series_data[label] = [
                    {"time": datetime.fromtimestamp(t).isoformat(), "count": c}
                    for t, c in sorted(buckets.items())
                ]
        
        response_data = {
            'current_detections': dict(current_detections),
            'time_series': time_series_data,
            'total_objects': sum(current_detections.values()),
            'active_tracks': len([tid for tid, traj in object_trajectories.items() 
                                if traj and current_time - traj[-1][2] < 5]),
            'device': device,
            'model_type': 'YOLOv11',
            'zones': detection_zones
        }
        
    return jsonify(response_data)

@app.route('/alerts')
def get_alerts():
    with analytics_lock:
        # Return recent alerts
        recent_alerts = list(alert_history)[-20:]  # Last 20 alerts
    return jsonify(recent_alerts)

@app.route('/config', methods=['GET', 'POST'])
def config():
    from flask import request
    
    if request.method == 'POST':
        config_data = request.get_json()
        # Update configuration
        global CONFIDENCE_THRESHOLD, ALERT_THRESHOLD, detection_zones
        
        if 'confidence_threshold' in config_data:
            CONFIDENCE_THRESHOLD = float(config_data['confidence_threshold'])
        
        if 'alert_thresholds' in config_data:
            ALERT_THRESHOLD.update(config_data['alert_thresholds'])
        
        if 'zones' in config_data:
            detection_zones = config_data['zones']
        
        return jsonify({'status': 'success'})
    
    # GET request - return current config
    config_data = {
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'alert_thresholds': ALERT_THRESHOLD,
        'zones': detection_zones,
        'tracking_enabled': TRACKING_ENABLED,
        'pose_estimation': POSE_ESTIMATION,
        'segmentation': SEGMENTATION
    }
    return jsonify(config_data)

if __name__ == "__main__":
    print("🚀 Starting Advanced YOLOv11 Video Analytics Dashboard...")
    print(f"💡 Device: {device}")
    print(f"🎯 Model: YOLOv11 {'with Pose' if POSE_ESTIMATION else ''} {'with Segmentation' if SEGMENTATION else ''}")
    print(f"📊 Features: Tracking={TRACKING_ENABLED}, Zones={ZONE_DETECTION}")
    
    # Start background threads
    threading.Thread(target=video_capture_thread, daemon=True, name="VideoCapture").start()
    threading.Thread(target=analytics_thread, daemon=True, name="Analytics").start()
    
    print("🎯 Dashboard running at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)