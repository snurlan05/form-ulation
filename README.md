# Form-ulation — AI Bicep Curl Form Analyzer

Form-ulation is a real-time AI personal trainer that uses computer vision and machine learning to analyze your bicep curl form. It detects bad form instantly — like swinging your back, flaring your elbows, or shrugging your shoulders — and gives you live feedback while you exercise.

---

## How It Works

The system is built around a full ML pipeline:

```
Videos (good/bad form) → Pose Estimation (YOLOv8) → Feature Extraction → Random Forest Classifier → Live Feedback
```

### 1. Data Collection (`collect_data.py`)
Raw workout videos are processed through **YOLOv8 Pose** estimation to extract skeleton keypoints for every frame. The script:
- Detects all people in the frame and focuses on the **largest bounding box** (the main subject)
- Extracts 17 body keypoints (34 normalized x/y values) using YOLOv8n-pose
- Computes **7 engineered biomechanical features** per frame:

| Feature | Description |
|---|---|
| `elbow_offset` | Horizontal distance of elbow from shoulder (elbow flare) |
| `back_lean` | Angle of the spine relative to vertical (back swing) |
| `wrist_velocity` | Speed of wrist movement (momentum/cheating) |
| `shoulder_shrug` | Vertical gap between ear and shoulder |
| `elbow_flare` | Elbow displacement from torso |
| `neck_angle` | Angle formed by ear, shoulder, and hip |
| `rep_phase` | Whether curling up (+1), lowering (-1), or holding (0) |

Data is saved to `batch_good_data.csv` and `batch_bad_data.csv` with labels `1` (good) and `0` (bad).

### 2. Model Training (`train_final.py`)
Training data is pulled from **Snowflake** (cloud data warehouse), then locally:
- Features are **standardized** with `StandardScaler`
- A **Random Forest Classifier** (300 trees, `class_weight='balanced'`) is trained on an 80/20 train/test split
- The trained model, scaler, and feature column list are bundled into **`bicep_model_v2.pkl`** via `joblib`

### 3. Live Inference
Two deployment modes are available:

#### Terminal / OpenCV Demo (`live_demo.py`)
Runs directly from your webcam using OpenCV. Loads the trained model and YOLOv8, processes each camera frame in real time, and overlays a **color-coded bounding box** with feedback right on the video window:
- 🟩 **Green box** = GOOD FORM + confidence %
- 🟥 **Red box** = BAD FORM + confidence %

A **cooldown timer** (30 frames) prevents the feedback from flickering between consecutive bad frames.

#### Streamlit Web App (`App.py`)
The polished, browser-based version. Uses `streamlit-webrtc` to stream your webcam through the browser, running the same YOLOv8 + Random Forest pipeline inside a `VideoTransformerBase` class. Provides a two-column layout with the live video feed on the left and a metrics guide on the right.

---

## Frontend — React Web UI (`/frontend`)

Built with **React 19 + Vite**, the frontend provides a web-based alternative to the Streamlit app:

| Component | Purpose |
|---|---|
| `Recorder.jsx` | Accesses the webcam via `getUserMedia`, lets the user record a clip, and shows a replay preview |
| `ChatPanel.jsx` | Displays the AI coaching response as a chat-bubble conversation |
| `ScorePanel.jsx` | Shows a numeric form quality score |
| `VideoLinks.jsx` | Renders curated tutorial video links returned by the backend |
| `client.js` | Sends the recorded video clip as a `multipart/form-data` POST to `/analyze` |

The frontend records a workout clip in **WebM/VP9** format, uploads it to the backend `/analyze` endpoint, and renders:
- A **form quality score** (0–100)
- A **feature breakdown** of what was good or bad
- **AI coaching messages** from the backend
- **Tutorial video links** for improvement

---

## Getting Started

### Prerequisites

- Python 3.12
- Node.js 20+ and npm (frontend only)
- A webcam (Streamlit app and React Recorder)
- No Snowflake credentials are needed to *run* the app — only to
  re-train the model (`train_final.py`).

### 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> The full install includes PyTorch (~900 MB wheel) for YOLOv8. To run
> only the FastAPI backend, `pip install -r backend/requirements.txt`
> installs the same ML stack without Streamlit/Snowflake.

### 2a. FastAPI backend + React frontend (full-stack, recommended)

```bash
# Terminal 1 — backend on http://localhost:8000
uvicorn backend.app:app --port 8000
# Check it: curl http://localhost:8000/health

# Terminal 2 — frontend on http://localhost:5173
cd frontend
npm install
npm run dev
```

Record a clip in the browser, press **Analyze**, and the backend
returns a 0–100 form score, a feature breakdown, coach messages, and
tutorial links. `POST /analyze` accepts `multipart/form-data`
(field `video`, WebM/MP4/MOV/AVI, max 100 MB). It analyses every 2nd
frame (max 600 analysed frames), runs the same YOLOv8 + Random Forest
pipeline as the Streamlit app, and returns 422 if no person is
detected in any analysed frame. API docs: http://localhost:8000/docs

### 2b. Streamlit web app (alternative interface)

```bash
streamlit run App.py
# http://localhost:8501 — live webcam analysis in the browser
```

### 2c. Terminal webcam demo

```bash
python live_demo.py     # OpenCV window; press 'q' to quit.
# Needs a display and the GUI build of OpenCV (opencv-python, not
# opencv-python-headless). The Streamlit and FastAPI paths do not.
```

### 3. Re-training (optional)

```bash
cp .env.example .env    # fill in SNOW_USER / SNOW_PASS / SNOW_ACCOUNT
python train_final.py   # downloads data from Snowflake, trains, writes bicep_model_v2.pkl
python collect_data.py  # extract features from videos in good_videos/ or bad_videos/
```

---

## Model files

| File | Status |
|---|---|
| `bicep_model_v2.pkl` (11 MB) | **Canonical model** — dict `{model, scaler, features}`: Random Forest (300 trees) + StandardScaler + 41 feature names. Used by `App.py`, `live_demo.py`, and `backend/`. |
| `yolov8n-pose.pt` (6.6 MB) | YOLOv8 nano pose weights. Required at runtime; downloaded by Ultralytics if missing. |
| `bicep_model.pkl` (5.8 MB) | Older model, superseded by v2. Bare Random Forest with 125 input features and no bundled scaler/columns — **not used by any current code**. Proposed: `git rm --cached` after confirming v2 (see below). |
| `scaler.pkl` + `model_columns.pkl` | Older scaler (125 features) and column list (31 features) — they do not match each other or the v1 model, and are **not used by any current code**. Proposed: `git rm --cached`. |
| `batch_good_data.csv` (7.0 MB) + `batch_bad_data.csv` (5.9 MB) | Training frames extracted by `collect_data.py` (19,056 rows). They are the only copy of the labelled training data, so they are **kept on disk**; proposed: `git rm --cached` and store them outside Git (or Git LFS) rather than deleting them. |

No held-out test accuracy is recorded for the v2 model. `train_final.py`
printed an accuracy at train time but did not save it, and scoring the
model on the committed CSVs gives 100% only because those CSVs *are*
the training data — that number is not a generalisation claim and is
not quoted as one.

## Project Structure

```
form-ulation/
├── collect_data.py        # Step 1: Extract poses from training videos → CSV
├── train_final.py         # Step 2: Train Random Forest from Snowflake data
├── live_demo.py           # Step 3a: Live webcam demo (OpenCV window)
├── App.py                 # Step 3b: Live webcam demo (Streamlit web app)
├── requirements.txt       # Full Python dependencies (converted from packages.txt)
├── .env.example           # Snowflake vars for training (no real values)
│
├── backend/               # Step 3c: FastAPI backend for the React frontend (new)
│   ├── app.py             #   FastAPI app: GET /health, POST /analyze
│   ├── pose.py            #   Feature extraction (same logic as App.py)
│   ├── coaching.py        #   Score / breakdown / coach messages / links
│   └── requirements.txt   #   Backend-only dependencies
│
├── bicep_model_v2.pkl     # Trained model bundle (model + scaler + features)
├── yolov8n-pose.pt        # YOLOv8 nano pose estimation model weights
│
├── batch_good_data.csv    # Labeled training frames — good form (label=1)
├── batch_bad_data.csv     # Labeled training frames — bad form (label=0)
│
└── frontend/              # React + Vite web UI
    └── src/
        ├── App.jsx
        ├── components/
        │   ├── Recorder.jsx
        │   ├── ChatPanel.jsx
        │   ├── ScorePanel.jsx
        │   └── VideoLinks.jsx
        └── api/
            └── client.js  # POST /analyze (multipart field "video")
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Pose Estimation | YOLOv8n-pose (Ultralytics) |
| ML Classifier | Random Forest (scikit-learn) |
| Live Video | OpenCV / streamlit-webrtc |
| Backend UI | Streamlit |
| Frontend UI | React 19 + Vite |
| Backend API | FastAPI + Uvicorn (`backend/`, `POST /analyze`) |
| Cloud Data | Snowflake |
| Model Storage | joblib (`.pkl`) |

---

## Key Design Decisions

- **YOLOv8 over MediaPipe** — YOLOv8 handles multi-person scenes naturally and selects the largest detected person as the subject, making it robust to crowded gym environments.
- **Engineered features on top of raw keypoints** — Simply feeding 34 raw keypoint coordinates into the classifier wasn't enough. The biomechanical features (back lean, wrist velocity, rep phase, etc.) encode domain knowledge about *what bad form looks like*, dramatically improving accuracy.
- **Bad-form cooldown** — Rather than flashing "BAD FORM" for a single anomalous frame, feedback persists for 20–30 frames after a bad prediction, which matches how long a bad-form rep typically plays out.
- **Snowflake for training data** — Centralized cloud storage enables collaborative labeling and easy scaling of the dataset without committing large CSVs to Git.

---

## Limitations (honest notes)

- **No verified held-out accuracy.** The v2 model was trained on the
  committed CSVs and no held-out test score was saved. Do not quote an
  accuracy % for this model until it is measured on unseen videos.
- **The score does not generalise to new people/setups yet (verified
  2026-11-11).** A real 6.5 s curl clip of a different person (Wikimedia
  Commons "CURL BÍCEPS", rear three-quarter view, animated subject) was
  analysed end-to-end by the new backend: 78/78 frames had a person
  detected and the pipeline ran cleanly, but the model classified all 78
  frames as bad form (score 0/100, mean confidence 75%), even though the
  engineered features looked good (back lean 2.4°, elbow flare 0.04 —
  both at/better than the good-form training means). The original
  `live_demo.py` logic gives the identical 0/78 on the same clip, so
  the backend is faithful — the model itself is the limit. It was
  trained on one subject/setup (100% on its own training CSVs) and its
  top features are raw keypoint coordinates (K23, K25), so it has
  effectively memorised that subject's proportions and framing. Treat
  the score as meaningful only for clips recorded in conditions similar
  to the training videos. Future work: collect labelled clips from
  several people/angles, hold out a test set, and consider dropping raw
  keypoints in favour of the engineered features only.
- **Left arm only.** Pose features use the left-side keypoints
  (shoulder 5, elbow 7, wrist 9, hip 11, ear 3). A right-arm-only curl
  facing away from the camera may be misread.
- **One person at a time.** The largest detected person is analysed;
  other people in frame are ignored.
- **Rep count is an estimate.** The backend counts reps from elbow-angle
  crossings (extended → flexed → extended); it is a heuristic, not a
  validated rep counter.
- **Coach messages are rule-based, not an LLM.** They compare feature
  means on bad-form vs good-form frames and name the worst issues.
- **Video only, no live React analysis.** The React frontend records a
  clip and uploads it for analysis; live per-frame feedback is only in
  the Streamlit / OpenCV demos.
- **`/analyze` caps work per upload** at 600 analysed frames (every 2nd
  frame) and 100 MB, to bound inference time on CPU.
