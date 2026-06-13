import gradio as gr
from ultralytics import YOLO
import numpy as np

# Downloads automatically on first run
model = YOLO('yolov8n.pt')

def detect(image):
    if image is None:
        return None
    results = model(image, verbose=False)
    return results[0].plot()

gr.Interface(
    fn=detect,
    inputs=gr.Image(sources=["webcam"], streaming=True),
    outputs=gr.Image(),
    title="Real-time Camera Detection",
    description="Point your camera at objects to detect them using YOLOv8",
    live=True
).launch()
