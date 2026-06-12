from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import logging
from collections import Counter
import time
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Helper function to convert numpy types to Python types
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

# Load YOLOv8n model
try:
    model = YOLO('yolov8n.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info(f"Model loaded successfully on {device}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Store detection results
detection_results = {
    'objects': [],
    'count': 0,
    'last_update': time.time(),
    'object_counter': Counter()
}

class Camera:
    def __init__(self):
        self.camera = None
        self.is_running = False
        
    def start(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Could not open camera")
                return False
            self.is_running = True
            return True
        return False
    
    def stop(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.is_running = False
            
    def get_frame(self):
        if self.camera and self.is_running:
            success, frame = self.camera.read()
            if success:
                return frame
        return None

camera_manager = Camera()

def process_frame(frame):
    """Process frame with YOLO detection and return results"""
    global detection_results
    
    if model is None:
        return frame, []
    
    # Run YOLO detection
    results = model(frame, stream=False)
    
    detected_objects = []
    
    # Draw bounding boxes
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get coordinates - convert to Python int
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])  # Convert to float
                class_id = int(box.cls[0])  # Convert to int
                class_name = model.names[class_id]
                
                # Store detected object with converted types
                detected_objects.append({
                    'name': class_name,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]  # Ensure int
                })
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with background
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Update detection results
    if detected_objects:
        detection_results['objects'] = detected_objects
        detection_results['count'] = len(detected_objects)
        detection_results['last_update'] = time.time()
        detection_results['object_counter'] = Counter([obj['name'] for obj in detected_objects])
    
    return frame, detected_objects

def generate_frames():
    """Generate video frames for streaming"""
    while True:
        frame = camera_manager.get_frame()
        if frame is None:
            break
            
        # Process frame for detection
        processed_frame, detected_objects = process_frame(frame)
        
        # Add detection info overlay on frame
        if detected_objects:
            y_offset = 30
            cv2.putText(processed_frame, f"Detected: {len(detected_objects)} objects", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 25
            # Show top 3 objects
            for obj in detected_objects[:3]:
                cv2.putText(processed_frame, f"- {obj['name']}: {obj['confidence']:.2f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 20
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera')
def start_camera():
    if camera_manager.start():
        return jsonify({"status": "success", "message": "Camera started"})
    return jsonify({"status": "error", "message": "Could not start camera"})

@app.route('/stop_camera')
def stop_camera():
    camera_manager.stop()
    # Clear results
    global detection_results
    detection_results = {
        'objects': [],
        'count': 0,
        'last_update': time.time(),
        'object_counter': Counter()
    }
    return jsonify({"status": "success", "message": "Camera stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_results')
def get_results():
    """API endpoint to get detection results with proper JSON serialization"""
    global detection_results
    
    # Convert detection results to serializable format
    serializable_results = {
        'status': 'success',
        'results': {
            'total_objects': int(detection_results['count']),  # Convert to int
            'objects': convert_to_serializable(detection_results['objects'][:10]),
            'object_summary': {k: int(v) for k, v in detection_results['object_counter'].items()},  # Convert counts to int
            'timestamp': float(detection_results['last_update'])  # Convert to float
        }
    }
    
    return jsonify(serializable_results)

@app.route('/model_info')
def model_info():
    if model:
        return jsonify({
            "status": "success",
            "model": "YOLOv8n",
            "device": device,
            "classes": int(len(model.names))  # Convert to int
        })
    return jsonify({"status": "error", "message": "Model not loaded"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)