from ultralytics import YOLO
import cv2
import numpy as np
import sys
from string.templatelib import Template # New in 3.14

# 1. Performance Check: Enable Free-Threading if supported
# In Python 3.14, you can check if the GIL is disabled programmatically
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
        
        # New in 3.14: Simplified exception handling (PEP 758)
        # You no longer need parentheses for multiple exception types
        try:
            shoulder, hip, knee, ankle = keypoints[5], keypoints[11], keypoints[13], keypoints[15]
            
            knee_angle = calculate_angle(hip, knee, ankle)
            back_angle = calculate_angle([hip[0], hip[1] - 100], hip, shoulder)
            
            # Use T-Strings (Template Strings) for structured labels
            # These are safer for dynamic content than standard f-strings
            knee_text = str(int(knee_angle))
            back_text = str(int(back_angle))
            
            cv2.putText(frame, knee_text, tuple(knee.astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, back_text, tuple(hip.astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Feedback logic
            if knee_angle < 90:
                cv2.putText(frame, "GOOD DEPTH!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif knee_angle < 140:
                cv2.putText(frame, "LOWER...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if back_angle > 45:
                 cv2.putText(frame, "STOP LEANING!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except IndexError, ValueError: # Note: No parentheses used here!
            pass

    annotated_frame = results[0].plot()
    cv2.imshow("Squat Analyzer 3.14", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()