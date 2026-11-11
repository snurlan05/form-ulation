"""Form-ulation FastAPI backend — POST /analyze for the React frontend.

Run (from the repo root):
    pip install -r backend/requirements.txt
    uvicorn backend.app:app --port 8000

Then, in another terminal:
    cd frontend && npm install && npm run dev     # Vite proxies /analyze -> :8000

Pipeline (same as App.py / live_demo.py, no new model):
    uploaded video -> OpenCV frames -> YOLOv8n-pose keypoints (largest
    person) -> 7 engineered features + 34 keypoints -> StandardScaler ->
    Random Forest (bicep_model_v2.pkl) per frame -> aggregate into a
    0-100 score + coach messages + tutorial links (see coaching.py).

Model files are loaded once at startup. Paths can be overridden with:
    FORMULATION_MODEL_PATH, FORMULATION_YOLO_PATH
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.coaching import build_response
from backend.pose import (
    FrameState,
    estimate_reps,
    extract_frame_features,
    predict_frame,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("FORMULATION_MODEL_PATH", REPO_ROOT / "bicep_model_v2.pkl"))
YOLO_PATH = Path(os.getenv("FORMULATION_YOLO_PATH", REPO_ROOT / "yolov8n-pose.pt"))

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
FRAME_STRIDE = 2          # analyse every 2nd frame (~15 fps at 30 fps)
MAX_ANALYZED_FRAMES = 600  # cap on analysed frames per upload

app = FastAPI(title="Form-ulation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_models = {}


def load_models():
    """Load the Random Forest bundle + YOLO pose model once (cached)."""
    if _models:
        return _models
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model bundle not found: {MODEL_PATH}")
    if not YOLO_PATH.exists():
        raise RuntimeError(f"YOLO weights not found: {YOLO_PATH}")
    import torch
    from ultralytics import YOLO

    bundle = joblib.load(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _models.update(
        {
            "model": bundle["model"],
            "scaler": bundle["scaler"],
            "features": bundle["features"],
            "yolo": YOLO(str(YOLO_PATH)),
            "device": device,
        }
    )
    return _models


@app.get("/health")
def health():
    try:
        m = load_models()
        device = m["device"]
    except Exception as exc:  # models not loadable yet — report it, don't crash
        return {"status": "error", "detail": str(exc)}
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "yolo": YOLO_PATH.name,
        "device": device,
    }


@app.get("/")
def root():
    return {
        "name": "Form-ulation API",
        "docs": "/docs",
        "health": "/health",
        "analyze": "POST /analyze (multipart field 'video')",
    }


def analyze_video(path: str):
    """Run the App.py pipeline over a video file. Returns stats dict."""
    m = load_models()
    model, scaler, feature_cols = m["model"], m["scaler"], m["features"]
    yolo, device = m["yolo"], m["device"]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Could not open video file.")
    try:
        state = FrameState()
        frames_total = 0
        frames_analyzed = 0
        frames_with_person = 0
        frames_good = 0
        probabilities: list[float] = []
        all_features: list[dict] = []
        good_features: list[dict] = []
        bad_features: list[dict] = []
        elbow_angles: list[float] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_total += 1
            if frames_total % FRAME_STRIDE != 0:
                continue
            if frames_analyzed >= MAX_ANALYZED_FRAMES:
                break
            frames_analyzed += 1

            results = yolo(frame, device=device, verbose=False)
            if (
                results[0].keypoints is None
                or len(results[0].keypoints.xyn) == 0
                or results[0].boxes is None
                or len(results[0].boxes.xyxy) == 0
            ):
                continue

            # Largest person, same as App.py.
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            best_idx = int(np.argmax(areas))
            keypoints = results[0].keypoints.xyn[best_idx].cpu().numpy()
            if keypoints.shape[0] < 12:
                continue

            input_values, diag = extract_frame_features(keypoints, state)
            prediction, probability = predict_frame(
                model, scaler, feature_cols, input_values
            )

            frames_with_person += 1
            probabilities.append(probability)
            all_features.append(diag)
            elbow_angles.append(diag["elbow_angle"])
            if prediction == 1:
                frames_good += 1
                good_features.append(diag)
            else:
                bad_features.append(diag)
    finally:
        cap.release()

    if frames_total == 0:
        raise HTTPException(
            status_code=422, detail="Video contains no readable frames."
        )
    if frames_with_person == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "No person was detected in any analysed frame. "
                "Re-record with your full upper body in frame and good lighting."
            ),
        )

    return {
        "frames_total": frames_total,
        "frames_analyzed": frames_analyzed,
        "frames_with_person": frames_with_person,
        "frames_good": frames_good,
        "frames_bad": frames_with_person - frames_good,
        "mean_probability": float(np.mean(probabilities)) if probabilities else 0.0,
        "reps_estimated": estimate_reps(elbow_angles),
        "all_frame_features": all_features,
        "good_frame_features": good_features,
        "bad_frame_features": bad_features,
    }


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    ext = Path(video.filename or "").suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported video type '{ext}'. Use mp4, mov, avi, webm or mkv.",
        )
    contents = await video.read()
    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail="Video is larger than the 100 MB limit."
        )

    suffix = ext if ext in ALLOWED_EXTENSIONS else ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        try:
            stats = analyze_video(tmp_path)
        except (RuntimeError, ImportError) as exc:
            # Model files missing or ML deps not installed — a server
            # problem, not a bad upload. Report 503, not a 500 traceback.
            raise HTTPException(
                status_code=503, detail=f"Model unavailable: {exc}"
            ) from exc
    finally:
        os.unlink(tmp_path)
    return build_response(stats)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
