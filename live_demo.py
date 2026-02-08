import cv2
import numpy as np
import pandas as pd
import joblib
import torch
from ultralytics import YOLO

# --- 1. LOAD RANDOM FOREST MODEL ---
data = joblib.load("bicep_model_v2.pkl")
forest_model = data['model']
scaler = data['scaler']
feature_cols = data['features']

# --- 2. LOAD YOLOv8 POSE MODEL ---
yolo_model = YOLO('yolov8n-pose.pt')  # Just load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"👀 YOLO running on {device.upper()}")

# --- 3. UTILITY FUNCTIONS ---
def calculate_angle(a, b, c):
    """Calculate angle ABC in degrees"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# --- 4. STATE VARIABLES ---
prev_wrist_y = 0
prev_elbow_angle = 180
bad_form_cooldown = 0
current_message = "GOOD FORM"
current_color = (0, 255, 0)  # Green

# --- 5. CAMERA SETUP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("📷 Starting camera. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- 6. RUN POSE DETECTION ---
    results = yolo_model(frame, device=device, verbose=False)
    annotated_frame = results[0].plot()

    if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
        # --- Select largest person ---
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        best_idx = int(np.argmax(areas))

        keypoints = results[0].keypoints.xyn[best_idx].cpu().numpy()
        flat_keypoints = keypoints.flatten()

        # --- 7. CALCULATE FEATURES ---
        elbow_offset = abs(keypoints[7][0] - keypoints[5][0])
        vertical_ref = [keypoints[11][0], keypoints[11][1]-0.5]
        back_lean = calculate_angle(vertical_ref, keypoints[11], keypoints[5])

        current_wrist_y = keypoints[9][1]
        wrist_velocity = 0 if prev_wrist_y==0 else abs(current_wrist_y - prev_wrist_y) * 100
        prev_wrist_y = current_wrist_y

        shoulder_shrug = abs(keypoints[3][1] - keypoints[5][1])
        elbow_flare = abs(keypoints[7][0] - keypoints[5][0])
        neck_angle = calculate_angle(keypoints[3], keypoints[5], keypoints[11])

        current_elbow_angle = calculate_angle(keypoints[5], keypoints[7], keypoints[9])
        angle_change = current_elbow_angle - prev_elbow_angle
        rep_phase = 1 if angle_change < -1 else (-1 if angle_change > 1 else 0)
        prev_elbow_angle = current_elbow_angle

        # --- 8. PREPARE INPUT FOR RANDOM FOREST ---
        input_values = [
            elbow_offset, back_lean, wrist_velocity,
            shoulder_shrug, elbow_flare, neck_angle, rep_phase
        ] + flat_keypoints.tolist()

        input_df = pd.DataFrame([input_values], columns=feature_cols)
        input_scaled = scaler.transform(input_df)

        # --- 9. MAKE PREDICTION ---
        prediction = forest_model.predict(input_scaled)[0]
        probability = forest_model.predict_proba(input_scaled)[0][prediction]

        # --- 10. UPDATE MESSAGE WITH COOLDOWN ---
        if prediction == 0:  # BAD FORM
            bad_form_cooldown = 30
            current_message = f"BAD FORM ({probability*100:.0f}%)"
            current_color = (0, 0, 255)
        elif bad_form_cooldown > 0:
            bad_form_cooldown -= 1
        else:
            current_message = f"GOOD FORM ({probability*100:.0f}%)"
            current_color = (0, 255, 0)

        # --- 11. DRAW BOX + MESSAGE ---
        box = boxes[best_idx].astype(int)
        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), current_color, 4)
        cv2.rectangle(annotated_frame, (box[0], box[1]-40), (box[0]+350, box[1]), current_color, -1)
        cv2.putText(annotated_frame, current_message, (box[0]+5, box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- 12. SHOW FRAME ---
    cv2.imshow("Live", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
