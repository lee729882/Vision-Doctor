import cv2
import sys
import os
from ultralytics import YOLO
from pathlib import Path

def run_inference(image_path, model_path='models/best_cow_contamination.pt'):
    if not os.path.exists(model_path):
        print(f"Error: Model not found ({model_path})")
        return
    model = YOLO(model_path)
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Image load fail ({image_path})")
        return
    results = model.predict(source=img, conf=0.25, imgsz=640)
    for result in results:
        annotated_img = result.plot()
        is_contaminated = any(int(box.cls) == 80 for box in result.boxes)
        status = "CONTAMINATION DETECTED!" if is_contaminated else "CLEAN"
        print(f'--- [Report] ---')
        print(f'File: {os.path.basename(image_path)}')
        print(f'Status: {status}')
        print(f'Detections: {len(result.boxes)}')
        cv2.imshow("Vision Doctor - Inference System", annotated_img)
        print("\nPress any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_inference(sys.argv[1])
    else:
        root = Path(r'c:\Cattle-Vision Doctor')
        samples = list((root / 'data/images/val').glob('*.jpg'))
        if samples:
            run_inference(samples[0])
        else: print('No samples found.')
