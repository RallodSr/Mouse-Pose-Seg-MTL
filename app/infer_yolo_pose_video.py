"""
Run YOLO-Pose inference on a video file.

Usage:
    python app/infer_yolo_pose_video.py --input video.mp4 --weights runs/pose/train/weights/best.pt
"""
import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO


def run(input_path: str, weights: str, output_path: str) -> None:
    model = YOLO(weights)
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"Processing video: {input_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)

    cap.release()
    writer.release()
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--weights", required=True, help="Path to YOLO pose .pt file")
    parser.add_argument("--output", default="models/inference_output/yolo_pose_output.mp4")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run(args.input, args.weights, args.output)
