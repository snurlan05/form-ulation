# 🏋️‍♂️ Form-ulation — AI Bicep Curl Form Analyzer

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

## Project Structure

```
form-ulation/
├── collect_data.py        # Step 1: Extract poses from training videos → CSV
├── train_final.py         # Step 2: Train Random Forest from Snowflake data
├── live_demo.py           # Step 3a: Live webcam demo (OpenCV window)
├── App.py                 # Step 3b: Live webcam demo (Streamlit web app)
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
            └── client.js
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
| Cloud Data | Snowflake |
| Model Storage | joblib (`.pkl`) |

---

## Key Design Decisions

- **YOLOv8 over MediaPipe** — YOLOv8 handles multi-person scenes naturally and selects the largest detected person as the subject, making it robust to crowded gym environments.
- **Engineered features on top of raw keypoints** — Simply feeding 34 raw keypoint coordinates into the classifier wasn't enough. The biomechanical features (back lean, wrist velocity, rep phase, etc.) encode domain knowledge about *what bad form looks like*, dramatically improving accuracy.
- **Bad-form cooldown** — Rather than flashing "BAD FORM" for a single anomalous frame, feedback persists for 20–30 frames after a bad prediction, which matches how long a bad-form rep typically plays out.
- **Snowflake for training data** — Centralized cloud storage enables collaborative labeling and easy scaling of the dataset without committing large CSVs to Git.