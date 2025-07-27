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

# --- Configuration - Optimized for Low Latency ---
DETECTION_INTERVAL = 2  # Process every 2nd frame
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
FRAME_SKIP = 1
MAX_QUEUE_SIZE = 5  # Reduced queue size
TRACKING_ENABLED = True
POSE_ESTIMATION = False
SEGMENTATION = False

# Advanced analytics configuration
ANALYTICS_WINDOW = 300  # 5 minutes of data retention
ALERT_THRESHOLD = {'person': 5, 'car': 3}
ZONE_DETECTION = True

# Global variables with proper initialization
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
        current_time = time.time()
        tracked_results = []
        for det in detections:
            try:
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
            except Exception as e:
                logging.error(f"Error processing track: {e}")
                continue
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
    """Advanced object detection using YOLOv11 - Optimized for latency"""
    if model is None:
        return []
    try:
        # Convert OpenCV frame to PyTorch tensor
        img_tensor = torch.from_numpy(frame).to(device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0  # HWC -> CHW, normalize
        if use_half:
            img_tensor = img_tensor.half()

        # Run YOLOv11 inference with tracking
        if TRACKING_ENABLED:
            results = model.track(img_tensor, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
                                persist=True, verbose=False)
        else:
            results = model.predict(img_tensor, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        detected_objects = []
        current_time = time.time()

        # Iterate through results
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                try:
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

                        # Zone detection
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
                except Exception as e:
                    logging.error(f"Error processing detection {i}: {e}")
                    continue
        return detected_objects
    except Exception as e:
        logging.error(f"YOLOv11 detection error: {e}")
        return []

def generate_alerts(detections):
    alerts = []
    try:
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
    except Exception as e:
        logging.error(f"Error generating alerts: {e}")
    return alerts

def video_capture_thread():
    """Enhanced video capture thread optimized for low latency"""
    print("Starting webcam with low-latency optimizations...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer

    frame_count = 0
    fps_counter = deque(maxlen=30)
    last_log_time = time.time()

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam")
                time.sleep(0.01)
                continue

            frame_count += 1
            current_time = time.time()

            # Calculate FPS (for debugging)
            fps_counter.append(current_time)
            if len(fps_counter) > 1:
                fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
            else:
                fps = 0

            # Log FPS occasionally
            if current_time - last_log_time > 5:
                print(f"Camera Capture FPS: ~{fps:.2f}")
                last_log_time = current_time

            # Frame skipping based on interval
            if frame_count % DETECTION_INTERVAL != 0:
                continue

            # Add frame info for debugging (optional, can be removed for production)
            # cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(frame, f"YOLOv11", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add to processing queue
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                    frame_queue.put(frame, block=False)
                except (queue.Empty, queue.Full):
                    pass

        except Exception as e:
            logging.error(f"Error in video capture: {e}")
            time.sleep(0.05)
    cap.release()

def analytics_thread():
    """Enhanced analytics processing thread optimized for low latency"""
    global analytics_data, current_detections
    while True:
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            start_time = time.time()

            # Perform YOLOv11 detection
            detections = detect_objects_yolov11(frame)

            # Generate alerts
            alerts = generate_alerts(detections)

            # Update analytics data
            current_time = time.time()
            with analytics_lock:
                cutoff_time = current_time - ANALYTICS_WINDOW
                for key in list(analytics_data.keys()):
                    analytics_data[key] = [(t, v) for t, v in analytics_data[key] if t > cutoff_time]
                current_detections.clear()
                for det in detections:
                    label = det['label']
                    current_detections[label] += 1
                    analytics_data[label].append((current_time, 1))
                for alert in alerts:
                    alert_history.append(alert)

            # Draw enhanced visualizations
            annotated_frame = frame.copy()

            # Draw detection zones
            if ZONE_DETECTION:
                for zone in detection_zones:
                    if zone['active']:
                        try:
                            pts = np.array(zone['polygon'], np.int32)
                            cv2.polylines(annotated_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                            cv2.putText(annotated_frame, zone['name'],
                                      tuple(zone['polygon'][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        except Exception as e:
                            logging.warning(f"Error drawing zone {zone['name']}: {e}")

            # Draw detections with enhanced info
            for det in detections:
                try:
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
                            pts = np.array([(int(p[0]), int(p[1])) for p in trajectory], np.int32)
                            if len(pts) > 1:
                                pts = pts.reshape((-1, 1, 2))
                                cv2.polylines(annotated_frame, [pts], isClosed=False, color=color, thickness=2)

                    # Draw pose keypoints if available
                    if POSE_ESTIMATION and 'keypoints' in det:
                        keypoints = det['keypoints']
                        for kp in keypoints:
                            if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
                                cv2.circle(annotated_frame, (int(kp[0]), int(kp[1])), 3, (255, 255, 0), -1)

                except Exception as e:
                    logging.warning(f"Error drawing detection: {e}")

            # Performance info (optional, can be removed for production)
            processing_time = time.time() - start_time
            # cv2.putText(annotated_frame, f"PTime: {processing_time*1000:.1f}ms",
            #            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(annotated_frame, f"Dets: {len(detections)}",
            #            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Add to output queue
            try:
                processed_frame_queue.put(annotated_frame, block=False)
            except queue.Full:
                try:
                    processed_frame_queue.get_nowait()
                    processed_frame_queue.put(annotated_frame, block=False)
                except (queue.Empty, queue.Full):
                    pass

            frame_queue.task_done()

        except Exception as e:
            logging.error(f"Error in analytics thread: {e}")
            frame_queue.task_done()

# Flask Routes
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
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])  # Reduced JPEG quality
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
    try:
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
    except Exception as e:
        logging.error(f"Error in analytics endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/alerts')
def get_alerts():
    try:
        with analytics_lock:
            recent_alerts = list(alert_history)[-20:]
        return jsonify(recent_alerts)
    except Exception as e:
        logging.error(f"Error in alerts endpoint: {e}")
        return jsonify({'error': str(e)}), 500

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

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'model_loaded': model is not None,
            'device': device,
            'queue_sizes': {
                'frame_queue': frame_queue.qsize(),
                'processed_frame_queue': processed_frame_queue.qsize()
            },
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Low-Latency YOLOv11 Video Analytics Dashboard...")
    # Initialize model
    try:
        # Load different YOLOv11 models based on requirements
        if POSE_ESTIMATION:
            model = YOLO('yolo11n-pose.pt')
            print("✓ YOLOv11 Pose model loaded")
        elif SEGMENTATION:
            model = YOLO('yolo11n-seg.pt')
            print("✓ YOLOv11 Segmentation model loaded")
        else:
            model = YOLO('yolo11n.pt')
            print("✓ YOLOv11 Detection model loaded")

        # Optimize model for inference
        model.fuse()
        model.eval()

        # Enable GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # Enable half precision if using CUDA
        use_half = False
        if device != 'cpu':
            try:
                model.half()
                use_half = True
                print("✓ FP16 inference enabled")
            except Exception as e:
                print(f"- FP16 not enabled: {e}")

        print(f"✓ Using device: {device}")
    except Exception as e:
        print(f"✗ Error loading YOLOv11 model: {e}")
        print("Please install ultralytics: pip install ultralytics")
        exit(1)

    # Start background threads
    try:
        threading.Thread(target=video_capture_thread, daemon=True, name="VideoCapture").start()
        threading.Thread(target=analytics_thread, daemon=True, name="Analytics").start()
        print("✓ Background threads started")
    except Exception as e:
        print(f"✗ Error starting threads: {e}")

    print("🎯 Dashboard running at: http://localhost:5000")
    print("🏥 Health check available at: http://localhost:5000/health")

    # Run Flask app
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except Exception as e:
        print(f"✗ Error starting Flask app: {e}")