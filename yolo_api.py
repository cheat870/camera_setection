from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import torch
import os

app = Flask(__name__)
CORS(app)  # Allow all origins

# Load YOLO model
print("🚀 Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"✅ Model loaded on {device}")

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'model': 'YOLOv8n',
        'device': device,
        'endpoints': {
            '/detect': 'POST - Send image for detection',
            '/health': 'GET - Check API status',
            '/info': 'GET - Model information'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': device
    })

@app.route('/info')
def info():
    return jsonify({
        'model': 'YOLOv8n',
        'device': device,
        'num_classes': len(model.names),
        'classes': list(model.names.values())[:30]
    })

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get image from request
        data = request.json
        image_data = data.get('image', '')
        
        # Remove header if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run YOLO detection
        results = model(img_array, conf=0.25)
        
        # Parse results
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                detections.append({
                    'class': class_name,
                    'confidence': round(confidence, 3),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'device': device
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
