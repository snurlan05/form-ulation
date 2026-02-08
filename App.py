import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO
import torch

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Bicep Coach", layout="wide")
st.title("🏋️‍♂️ AI Bicep Curl Form Analyzer")
st.markdown("Ensure your full upper body is visible in the frame.")

# --- CACHE MODELS (So they don't reload every frame) ---
@st.cache_resource
def load_models():
    # Load your Random Forest logic
    data = joblib.load("bicep_model_v2.pkl")
    yolo = YOLO('yolov8n-pose.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return data, yolo, device

data, yolo_model, device = load_models()
forest_model = data['model']
scaler = data['scaler']
feature_cols = data['features']

# --- SHARED UTILITIES ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# --- VIDEO PROCESSING CLASS ---
class PoseTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_wrist_y = 0
        self.prev_elbow_angle = 180
        self.bad_form_cooldown = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Run YOLO
        results = yolo_model(img, device=device, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
            # Select largest person
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            best_idx = int(np.argmax(areas))

            keypoints = results[0].keypoints.xyn[best_idx].cpu().numpy()
            flat_keypoints = keypoints.flatten()

            # 2. Feature Engineering (Your Logic)
            elbow_offset = abs(keypoints[7][0] - keypoints[5][0])
            vertical_ref = [keypoints[11][0], keypoints[11][1]-0.5]
            back_lean = calculate_angle(vertical_ref, keypoints[11], keypoints[5])

            current_wrist_y = keypoints[9][1]
            wrist_velocity = 0 if self.prev_wrist_y == 0 else abs(current_wrist_y - self.prev_wrist_y) * 100
            self.prev_wrist_y = current_wrist_y

            shoulder_shrug = abs(keypoints[3][1] - keypoints[5][1])
            elbow_flare = abs(keypoints[7][0] - keypoints[5][0])
            neck_angle = calculate_angle(keypoints[3], keypoints[5], keypoints[11])

            current_elbow_angle = calculate_angle(keypoints[5], keypoints[7], keypoints[9])
            angle_change = current_elbow_angle - self.prev_elbow_angle
            rep_phase = 1 if angle_change < -1 else (-1 if angle_change > 1 else 0)
            self.prev_elbow_angle = current_elbow_angle

            # 3. Predict
            input_values = [
                elbow_offset, back_lean, wrist_velocity,
                shoulder_shrug, elbow_flare, neck_angle, rep_phase
            ] + flat_keypoints.tolist()

            input_df = pd.DataFrame([input_values], columns=feature_cols)
            input_scaled = scaler.transform(input_df)
            prediction = forest_model.predict(input_scaled)[0]
            probability = forest_model.predict_proba(input_scaled)[0][prediction]

            # 4. Determine UI State
            if prediction == 0:
                self.bad_form_cooldown = 20
                msg, color = f"BAD FORM ({probability*100:.0f}%)", (0, 0, 255)
            elif self.bad_form_cooldown > 0:
                self.bad_form_cooldown -= 1
                msg, color = "BAD FORM (HOLDING...)", (0, 0, 255)
            else:
                msg, color = f"GOOD FORM ({probability*100:.0f}%)", (0, 255, 0)

            # 5. Draw
            box = boxes[best_idx].astype(int)
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 4)
            cv2.putText(annotated_frame, msg, (box[0], box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return annotated_frame

# --- UI LAYOUT ---
col1, col2 = st.columns([3, 1])

with col1:
    webrtc_streamer(key="bicep-curl", video_transformer_factory=PoseTransformer)

with col2:
    st.subheader("Metrics Guide")
    st.info("**Good Form:** Elbows tucked, neutral back, controlled motion.")
    st.warning("**Bad Form:** Swinging back, elbow flare, or shrugging.")
    
    if st.button("Reset Session"):
        st.rerun()