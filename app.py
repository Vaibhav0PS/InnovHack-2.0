import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
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

# --- Configuration ---
DETECTION_INTERVAL = 5  # Process every 5th frame
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
MAX_QUEUE_SIZE = 10
TRACKING_ENABLED = True
POSE_ESTIMATION = False
SEGMENTATION = False

# Advanced analytics configuration
ANALYTICS_WINDOW = 300  # 5 minutes of data retention
ALERT_THRESHOLD = {'person': 10, 'car': 5}  # Example thresholds
ZONE_DETECTION = True

# Global variables
analytics_data = defaultdict(list)
current_detections = defaultdict(int)
object_trajectories = defaultdict(lambda: deque(maxlen=50))
alert_history = deque(maxlen=100)
analytics_lock = threading.Lock()

# Detection zones
detection_zones = [
    {"name": "Entry Zone", "polygon": [(50, 50), (200, 50), (200, 200), (50, 200)], "active": True},
    {"name": "Restricted Area", "polygon": [(300, 100), (500, 100), (500, 300), (300, 300)], "active": True}
]

# Model and device initialization
model = None
device = 'cpu'

def initialize_model():
    global model, device
    try:
        if POSE_ESTIMATION:
            model = YOLO('yolo11n-pose.pt')
            print("✓ YOLOv11 Pose model loaded")
        elif SEGMENTATION:
            model = YOLO('yolo11n-seg.pt')
            print("✓ YOLOv11 Segmentation model loaded")
        else:
            model = YOLO('yolo11n.pt')
            print("✓ YOLOv11 Detection model loaded")

        model.fuse()
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        if device != 'cpu':
            try:
                model.half()
                print("✓ FP16 inference enabled")
            except Exception as e:
                print(f"- FP16 not enabled: {e}")

        print(f"✓ Using device: {device}")
        return True
    except Exception as e:
        print(f"✗ Error loading YOLOv11 model: {e}")
        return False

# Thread-safe data structures
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
processed_frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
analytics_queue = queue.Queue(maxsize=100)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0

    def update_tracks(self, detections):
        current_time = time.time()
        tracked_results = []
        for det in detections:
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
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                object_trajectories[track_id].append((center_x, center_y, current_time))
        return tracked_results

def point_in_polygon(point, polygon):
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
    if model is None:
        return []
    try:
        img_tensor = torch.from_numpy(frame).to(device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if use_half:
            img_tensor = img_tensor.half()

        if TRACKING_ENABLED:
            results = model.track(img_tensor, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
                                persist=True, tracker="bytetrack.yaml", verbose=False)
        else:
            results = model.predict(img_tensor, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        detected_objects = []
        current_time = time.time()

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)

                for i in range(len(boxes)):
                    box = boxes[i]
                    conf = confidences[i]
                    cls = int(classes[i])
                    track_id = int(track_ids[i]) if track_ids[i] is not None else None
                    x1, y1, x2, y2 = box
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

                    zones_detected = []
                    if ZONE_DETECTION:
                        for zone in detection_zones:
                            if zone['active'] and point_in_polygon((center_x, center_y), zone['polygon']):
                                zones_detected.append(zone['name'])

                    detection_data = {
                        'label': model.names[cls],
                        'confidence': float(conf),
                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        'center': (center_x, center_y),
                        'track_id': track_id,
                        'zones': zones_detected,
                        'timestamp': current_time
                    }

                    detected_objects.append(detection_data)

        return detected_objects
    except Exception as e:
        logging.error(f"YOLOv11 detection error: {e}")
        return []

def generate_alerts(detections):
    alerts = []
    current_time = datetime.now()
    detection_counts = defaultdict(int)
    for det in detections:
        detection_counts[det['label']] += 1

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
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    fps_counter = deque(maxlen=30)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            time.sleep(0.1)
            continue

        frame_count += 1
        current_time = time.time()
        fps_counter.append(current_time)
        if len(fps_counter) > 1:
            fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])

        if frame_count % DETECTION_INTERVAL != 0:
            continue

        try:
            frame_queue.put(frame, block=False)
        except queue.Full:
            try:
                frame_queue.get_nowait()
                frame_queue.put(frame, block=False)
            except (queue.Empty, queue.Full):
                pass

    cap.release()

def analytics_thread():
    while True:
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        start_time = time.time()
        detections = detect_objects_yolov11(frame)
        alerts = generate_alerts(detections)

        with analytics_lock:
            cutoff_time = time.time() - ANALYTICS_WINDOW
            for key in list(analytics_data.keys()):
                analytics_data[key] = [(t, v) for t, v in analytics_data[key] if t > cutoff_time]
            current_detections.clear()
            for det in detections:
                label = det['label']
                current_detections[label] += 1
                analytics_data[label].append((time.time(), 1))
            for alert in alerts:
                alert_history.append(alert)

        annotated_frame = frame.copy()

        # Draw zones
        if ZONE_DETECTION:
            for zone in detection_zones:
                if zone['active']:
                    pts = np.array(zone['polygon'], np.int32)
                    cv2.polylines(annotated_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                    cv2.putText(annotated_frame, zone['name'],
                              tuple(zone['polygon'][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw detections
        for det in detections:
            x, y, w, h = det['bbox']
            label = det['label']
            conf = det['confidence']
            track_id = det['track_id']

            color = (0, 255, 0)
            if 'Restricted Area' in det['zones']:
                color = (0, 0, 255)
            elif det['zones']:
                color = (0, 255, 255)

            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            label_text = f"{label} {conf:.2f}"
            if track_id is not None:
                label_text += f" ID:{track_id}"
            cv2.putText(annotated_frame, label_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if track_id and track_id in object_trajectories:
                trajectory = list(object_trajectories[track_id])
                if len(trajectory) > 1:
                    pts = np.array([(int(p[0]), int(p[1])) for p in trajectory], np.int32)
                    if len(pts) > 1:
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [pts], isClosed=False, color=color, thickness=2)

        processing_time = time.time() - start_time
        cv2.putText(annotated_frame, f"PTime: {processing_time*1000:.1f}ms",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Dets: {len(detections)}",
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        try:
            processed_frame_queue.put(annotated_frame, block=False)
        except queue.Full:
            try:
                processed_frame_queue.get_nowait()
                processed_frame_queue.put(annotated_frame, block=False)
            except (queue.Empty, queue.Full):
                pass

        frame_queue.task_done()

@app.route('/')
def index():
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logging.error(f"Error rendering dashboard: {e}")
        return f"Dashboard template not found. Error: {e}", 500

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                if not processed_frame_queue.empty():
                    frame = processed_frame_queue.get(timeout=0.05)
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    time.sleep(0.01)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Video feed error: {e}")
                time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analytics')
def get_analytics():
    with analytics_lock:
        current_time = time.time()
        time_series_data = {}
        for label, data_points in analytics_data.items():
            if data_points:
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
            'model_type': 'YOLOv11' if model else 'No Model',
            'zones': detection_zones
        }
    return jsonify(response_data)

@app.route('/alerts')
def get_alerts():
    with analytics_lock:
        recent_alerts = list(alert_history)[-20:]
    return jsonify(recent_alerts)

@app.route('/config', methods=['GET', 'POST'])
def config():
    try:
        if request.method == 'POST':
            config_data = request.get_json()
            global CONFIDENCE_THRESHOLD, ALERT_THRESHOLD, detection_zones
            if 'confidence_threshold' in config_data:
                CONFIDENCE_THRESHOLD = float(config_data['confidence_threshold'])
            if 'alert_thresholds' in config_data:
                ALERT_THRESHOLD.update(config_data['alert_thresholds'])
            if 'zones' in config_data:
                detection_zones = config_data['zones']
            return jsonify({'status': 'success'})
        config_data = {
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'alert_thresholds': ALERT_THRESHOLD,
            'zones': detection_zones,
            'tracking_enabled': TRACKING_ENABLED,
            'pose_estimation': POSE_ESTIMATION,
            'segmentation': SEGMENTATION
        }
        return jsonify(config_data)
    except Exception as e:
        logging.error(f"Error in config endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Advanced YOLOv11 Video Analytics Dashboard...")
    model_loaded = initialize_model()
    if not model_loaded:
        print("⚠️  Running without YOLOv11 model - limited functionality")
    print(f"💡 Device: {device}")
    print(f"🎯 Model: {'YOLOv11' if model else 'No Model'} {'with Pose' if POSE_ESTIMATION else ''} {'with Segmentation' if SEGMENTATION else ''}")
    print(f"📊 Features: Tracking={TRACKING_ENABLED}, Zones={ZONE_DETECTION}")
    threading.Thread(target=video_capture_thread, daemon=True, name="VideoCapture").start()
    threading.Thread(target=analytics_thread, daemon=True, name="Analytics").start()
    print("🎯 Dashboard running at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)