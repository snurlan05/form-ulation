from ultralytics import YOLO
import cv2
import numpy as np
import sys

# 1. Performance Check: Python 3.14 Free-Threading check
if sys.version_info >= (3, 13) and hasattr(sys, '_is_gil_enabled'):
    gil_status = "Disabled" if not sys._is_gil_enabled() else "Enabled"
    print(f"GIL Status: {gil_status}")

# 2. Load YOLOv8
model = YOLO('yolov8n-pose.pt')

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, verbose=False)
    
    if results[0].keypoints is not None and results[0].keypoints.xy.nelement() > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        # New in 3.14: Simplified exception handling logic
        try:
            # --- CHANGE 1: Select Arm Joints instead of Leg Joints ---
            # 5 = Left Shoulder
            # 7 = Left Elbow
            # 9 = Left Wrist
            # 11 = Left Hip (Used for checking back sway)
            shoulder = keypoints[5]
            elbow = keypoints[7]
            wrist = keypoints[9]
            hip = keypoints[11]
            
            # --- CHANGE 2: Calculate Curl Angle ---
            # Angle at the Elbow (Shoulder -> Elbow -> Wrist)
            curl_angle = calculate_angle(shoulder, elbow, wrist)
            
            # Form Check: Torso Swing (Vertical -> Hip -> Shoulder)
            # Create a virtual vertical point above the hip
            vertical_point = [hip[0], hip[1] - 100]
            back_lean = calculate_angle(vertical_point, hip, shoulder)
            
            # Visualize the angles
            cv2.putText(frame, str(int(curl_angle)), tuple(elbow.astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # --- CHANGE 3: Bicep Specific Feedback Logic ---
            
            # 1. Check for Full Contraction (The Squeeze)
            # A good curl usually goes below 35-40 degrees
            if curl_angle < 35:
                cv2.putText(frame, "GOOD SQUEEZE!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            # 2. Check for Full Extension (The Stretch)
            # You want to go almost fully straight (approx 160-170 degrees)
            elif curl_angle > 160:
                cv2.putText(frame, "FULL EXTENSION", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            # 3. Check for Cheating (Swinging the back)
            # If torso moves more than 10 degrees back, you are using momentum
            if back_lean > 10:
                 cv2.putText(frame, "DONT SWING!", (50, 100), 
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        except IndexError, ValueError: # Using your requested style
            pass

    annotated_frame = results[0].plot()
    cv2.imshow("Bicep Analyzer 3.14", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()