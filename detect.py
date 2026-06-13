import os, sys, time, json, argparse
from pathlib import Path
from datetime import datetime
import numpy as np

if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'): np.bool = bool

from ultralytics import YOLO

def download_demo_image():
    import urllib.request
    os.makedirs("images", exist_ok=True)
    # Use direct stable URLs with no redirects
    urls = [
        "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?w=640",
        "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg?w=640",
        "https://farm2.staticflickr.com/1533/26541536141_41abe98db3_z.jpg",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            })
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
                if len(data) > 10000:  # must be at least 10KB
                    with open("images/demo.jpg", "wb") as f:
                        f.write(data)
                    print(f"✅ Demo image downloaded ({len(data)//1024}KB)")
                    return True
                else:
                    print(f"⚠️  File too small from {url}")
        except Exception as e:
            print(f"⚠️  Failed {url}: {e}")

    # Final fallback: generate a synthetic image with shapes YOLO can detect
    print("⚠️  All URLs failed — generating synthetic test image...")
    import cv2
    os.makedirs("images", exist_ok=True)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (135, 206, 235)  # sky blue background

    # Draw a person-like shape (rectangle = body)
    cv2.rectangle(img, (280, 150), (360, 400), (200, 150, 100), -1)  # body
    cv2.circle(img, (320, 120), 40, (200, 150, 100), -1)              # head

    # Draw a car-like shape
    cv2.rectangle(img, (50, 300), (230, 380), (50, 50, 200), -1)     # car body
    cv2.rectangle(img, (70, 260), (210, 310), (50, 50, 200), -1)     # roof
    cv2.circle(img, (90, 385), 20, (30, 30, 30), -1)                 # wheel
    cv2.circle(img, (190, 385), 20, (30, 30, 30), -1)                # wheel

    cv2.imwrite("images/demo.jpg", img)
    print("✅ Synthetic test image created")
    return True

def run_detection(source="images/", model_path="yolov8n.pt", conf=0.25, output_dir="results"):
    print(f"\n{'='*50}")
    print(f"  Camera Detection — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

    model = YOLO(model_path)
    print(f"✅ Model loaded: {len(model.names)} classes")

    source_path = Path(source)
    has_images = source_path.exists() and any(
        f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for f in source_path.glob("*")
    )

    if not has_images:
        print("No images found — fetching demo image...")
        download_demo_image()
        source = "images/"

    os.makedirs(output_dir, exist_ok=True)
    results = model.predict(
        source=source, conf=conf, save=True, save_txt=True,
        save_conf=True, project=output_dir, name="detection",
        exist_ok=True, verbose=False
    )

    total = 0
    all_dets = []
    for i, r in enumerate(results):
        boxes = r.boxes
        count = len(boxes) if boxes is not None else 0
        total += count
        dets = []
        if boxes is not None:
            for box in boxes:
                label = model.names[int(box.cls[0].item())]
                conf_val = float(box.conf[0].item())
                dets.append({"label": label, "confidence": round(conf_val, 3)})
                print(f"  ✅ {label:<20} {conf_val:.1%}")
        if not dets:
            print(f"  ⚪ Frame {i+1}: nothing detected")
        all_dets.append({"frame": i+1, "detections": dets})

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_frames": len(results),
        "total_objects": total,
        "results": all_dets
    }

    with open(f"{output_dir}/detection_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(f"{output_dir}/detection_log.txt", "w") as f:
        f.write(f"Run: {datetime.now()}\nFrames: {len(results)}\nObjects: {total}\n\n")
        for d in all_dets:
            f.write(f"Frame {d['frame']}:\n")
            for obj in d['detections']:
                f.write(f"  - {obj['label']} ({obj['confidence']:.1%})\n")
            if not d['detections']:
                f.write("  - nothing detected\n")

    print(f"\n{'='*50}")
    print(f"  Frames: {len(results)}  |  Objects: {total}")
    print(f"  Report: {output_dir}/detection_report.json")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="images/")
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--conf",   default=0.25, type=float)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run_detection(args.source, args.model, args.conf, args.output)
