import os, sys, time, json, argparse
from pathlib import Path
from datetime import datetime
import numpy as np

if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'): np.bool = bool

from ultralytics import YOLO

def run_detection(source="images/", model_path="yolov8n.pt", conf=0.25, output_dir="results"):
    print(f"\n{'='*50}")
    print(f"  Camera Detection — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

    model = YOLO(model_path)
    print(f"✅ Model loaded: {len(model.names)} classes")

    source_path = Path(source)
    if not source_path.exists() or not list(source_path.glob("*")):
        print("No images found — downloading demo image...")
        import urllib.request
        os.makedirs("images", exist_ok=True)
        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "images/demo.jpg")
        source = "images/"

    os.makedirs(output_dir, exist_ok=True)
    results = model.predict(source=source, conf=conf, save=True, save_txt=True,
                            save_conf=True, project=output_dir, name="detection",
                            exist_ok=True, verbose=False)

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
        all_dets.append({"frame": i+1, "detections": dets})

    report = {"timestamp": datetime.now().isoformat(),
              "total_frames": len(results), "total_objects": total, "results": all_dets}

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

    print(f"\n✅ Done! {len(results)} frames, {total} objects found")
    print(f"   Report: {output_dir}/detection_report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="images/")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run_detection(args.source, args.model, args.conf, args.output)
