from ultralytics import YOLO
import cv2
import csv
import os
import numpy as np

# --- 1. CONFIGURATION (ALREADY SET FOR BAD VIDEOS) ---
FOLDER_PATH = "bad_videos"       # <--- Looking in 'bad_videos' folder
OUTPUT_FILE = "batch_bad_data.csv"    # <--- Saving to 'batch_bad_data.csv'
LABEL = 0                        # <--- Label 0 means BAD FORM
# -----------------------------------------------------

# --- 2. SETUP ---
model = YOLO('yolov8n-pose.pt')

# Verify folder exists
if not os.path.exists(FOLDER_PATH):
    os.makedirs(FOLDER_PATH)
    print(f"ERROR: Folder '{FOLDER_PATH}' not found!")
    print("Please create a folder named 'bad_videos' and put your bad form videos inside it.")
    exit()

# Get list of all video files
video_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(('.mp4', '.mov'))]
total_videos = len(video_files)
print(f"--- FOUND {total_videos} BAD VIDEOS. STARTING BATCH PROCESS ---")

# --- MATH HELPER ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

# --- 3. PREPARE CSV ---
file_exists = os.path.exists(OUTPUT_FILE)

with open(OUTPUT_FILE, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        headers = ['label', 
                   'elbow_offset', 'back_lean', 'wrist_velocity', 
                   'shoulder_shrug', 'elbow_flare', 'neck_angle',
                   'rep_phase']
        headers += [f'k{i}' for i in range(34)]
        writer.writerow(headers)

# --- 4. MAIN LOOP ---
for i, video_file in enumerate(video_files):
    video_path = os.path.join(FOLDER_PATH, video_file)
    cap = cv2.VideoCapture(video_path)
    
    print(f"[{i+1}/{total_videos}] Processing: {video_file}...")

    # RESET HISTORY FOR NEW VIDEO
    prev_wrist_y = 0
    prev_elbow_angle = 180
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Run YOLO
        results = model(frame, verbose=False)
        
        # Draw the skeleton on the frame (Visuals)
        annotated_frame = results[0].plot()

        # --- FOCUS LOGIC: Find the Biggest Person ---
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            
            best_person_idx = 0
            max_area = 0
            
            # Loop through all detected boxes to find the largest one
            for idx, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_person_idx = idx

            # Draw "Target Locked" Box
            box = results[0].boxes.xyxy[best_person_idx].cpu().numpy().astype(int)
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4) # Red Box for Bad
            cv2.putText(annotated_frame, "BAD DATA TARGET", (box[0], box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Extract Keypoints for the TARGET person only
            if results[0].keypoints is not None:
                keypoints = results[0].keypoints.xyn[best_person_idx].cpu().numpy()
                flat_keypoints = keypoints.flatten()
                
                # --- PHYSICS CALCULATIONS ---
                elbow_offset = abs(keypoints[7][0] - keypoints[5][0]) 
                vertical_ref = [keypoints[11][0], keypoints[11][1] - 0.5]
                back_lean = calculate_angle(vertical_ref, keypoints[11], keypoints[5])
                
                current_wrist_y = keypoints[9][1]
                
                # Velocity Fix (Prevent first frame spike)
                if prev_wrist_y == 0: 
                    wrist_velocity = 0
                    prev_wrist_y = current_wrist_y
                else:
                    wrist_velocity = abs(current_wrist_y - prev_wrist_y) * 100 
                    prev_wrist_y = current_wrist_y 

                shoulder_shrug = abs(keypoints[3][1] - keypoints[5][1])
                elbow_flare = abs(keypoints[7][0] - keypoints[5][0])
                neck_angle = calculate_angle(keypoints[3], keypoints[5], keypoints[11])

                # Phase Logic
                current_elbow_angle = calculate_angle(keypoints[5], keypoints[7], keypoints[9])
                angle_change = current_elbow_angle - prev_elbow_angle
                
                rep_phase = 0 
                if angle_change < -1.0: rep_phase = 1 
                elif angle_change > 1.0: rep_phase = -1
                prev_elbow_angle = current_elbow_angle

                # --- SAVE ROW TO CSV (Using LABEL=0) ---
                data_row = [LABEL, 
                            elbow_offset, back_lean, wrist_velocity, 
                            shoulder_shrug, elbow_flare, neck_angle,
                            rep_phase] + flat_keypoints.tolist()
                
                with open(OUTPUT_FILE, mode='a', newline='') as f:
                    csv.writer(f).writerow(data_row)
                
                frame_count += 1

        # Show the video (Red Box)
        cv2.imshow("Batch Processor - BAD MODE", annotated_frame)
        
        # Press Q to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    cap.release()
    print(f"   -> Finished {video_file} ({frame_count} frames saved)")

cv2.destroyAllWindows()
print("--- ALL BAD VIDEOS PROCESSED ---")