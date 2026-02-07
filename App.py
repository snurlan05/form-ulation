from ultralytics import YOLO
import cv2
import numpy as np

# 1. Load YOLOv8
model = YOLO('yolov8n-pose.pt')

# 2. Math Function
def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Run AI
    results = model(frame, verbose=False)
    
    # 3. Logic: Extract Landmarks for Squat
    if results[0].keypoints is not None and results[0].keypoints.xy.nelement() > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        # YOLO Indices (Left Side): 5=Shoulder, 11=Hip, 13=Knee, 15=Ankle
        try:
            shoulder = keypoints[5]
            hip = keypoints[11]
            knee = keypoints[13]
            ankle = keypoints[15]
            
            # --- CALCULATION 1: KNEE ANGLE (Depth) ---
            # Angle between Hip -> Knee -> Ankle
            knee_angle = calculate_angle(hip, knee, ankle)
            
            # --- CALCULATION 2: BACK ANGLE (Posture) ---
            # We create an invisible point directly above the hip to represent "Vertical"
            vertical_point = [hip[0], hip[1] - 100] 
            # Angle between Vertical -> Hip -> Shoulder
            back_angle = calculate_angle(vertical_point, hip, shoulder)
            
            # --- VISUALIZATION ---
            
            # Draw Knee Angle (at the knee)
            cv2.putText(frame, str(int(knee_angle)), 
                        tuple(knee.astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw Back Angle (at the hip)
            cv2.putText(frame, str(int(back_angle)), 
                        tuple(hip.astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # --- FEEDBACK LOGIC ---
            
            # Check Depth
            if knee_angle < 90:
                cv2.putText(frame, "GOOD DEPTH!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif knee_angle < 140:
                cv2.putText(frame, "LOWER...", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            
            # Check Back Form (If you lean forward more than 45 degrees)
            if back_angle > 45:
                 cv2.putText(frame, "STOP LEANING!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except IndexError:
            pass

    # Show the video
    annotated_frame = results[0].plot()
    cv2.imshow("Squat Analyzer", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()