from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load YOLOv8n model
try:
    model = YOLO('yolov8n.pt')
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info(f"Model loaded successfully on {device}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

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
    """Process frame with YOLO detection"""
    if model is None:
        return frame
    
    # Run YOLO detection
    results = model(frame, stream=False)
    
    # Draw bounding boxes
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame

def generate_frames():
    """Generate video frames for streaming"""
    while True:
        frame = camera_manager.get_frame()
        if frame is None:
            break
            
        # Process frame for detection
        processed_frame = process_frame(frame)
        
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
    """Start camera stream"""
    if camera_manager.start():
        return jsonify({"status": "success", "message": "Camera started"})
    return jsonify({"status": "error", "message": "Could not start camera"})

@app.route('/stop_camera')
def stop_camera():
    """Stop camera stream"""
    camera_manager.stop()
    return jsonify({"status": "success", "message": "Camera stopped"})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/model_info')
def model_info():
    """Get model information"""
    if model:
        return jsonify({
            "status": "success",
            "model": "YOLOv8n",
            "device": device,
            "classes": len(model.names)
        })
    return jsonify({"status": "error", "message": "Model not loaded"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)