from ultralytics import YOLO
import cv2
import csv
import os
import numpy as np

# --- 1. SETUP ---
model = YOLO('yolov8n-pose.pt')
folder_path = "good_videos"  # <--- PUT YOUR VIDEOS IN THIS FOLDER
output_file = "batch_good_data.csv"

# Verify folder exists
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Created folder '{folder_path}'. Please put your videos inside it!")
    exit()

# Get list of all video files (MP4 and MOV)
video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.mov'))]
total_videos = len(video_files)

print(f"--- FOUND {total_videos} VIDEOS. STARTING BATCH PROCESS ---")

# --- MATH HELPERS ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

# --- 2. SETUP CSV ---
# Check if file exists so we don't overwrite headers if we run this twice
file_exists = os.path.exists(output_file)

with open(output_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    
    # Write headers only if file is new
    if not file_exists:
        headers = ['label', 
                   'elbow_offset', 'back_lean', 'wrist_velocity', 
                   'shoulder_shrug', 'elbow_flare', 'neck_angle',
                   'rep_phase']
        headers += [f'k{i}' for i in range(34)]
        writer.writerow(headers)

# --- 3. BATCH PROCESS LOOP ---
for i, video_file in enumerate(video_files):
    video_path = os.path.join(folder_path, video_file)
    cap = cv2.VideoCapture(video_path)
    
    print(f"[{i+1}/{total_videos}] Processing: {video_file}...")

    # RESET HISTORY FOR NEW VIDEO (Critical!)
    # We don't want velocity to jump from Video A to Video B
    prev_wrist_y = 0
    prev_elbow_angle = 180
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Run YOLO (Verbose=False to keep terminal clean)
        results = model(frame, verbose=False)

        # We still show the video so you can see it working
        # (Optional: Comment out imshow/waitKey to make it run 10x faster)
        annotated_frame = results[0].plot()
        cv2.imshow("Batch Processor", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Press Q to force quit early
            exit()

        if results[0].keypoints is not None and results[0].keypoints.xyn.nelement() > 0:
            keypoints = results[0].keypoints.xyn[0].cpu().numpy()
            flat_keypoints = keypoints.flatten()
            
            # --- PHYSICS CALCULATIONS ---
            elbow_offset = abs(keypoints[7][0] - keypoints[5][0]) 
            vertical_ref = [keypoints[11][0], keypoints[11][1] - 0.5]
            back_lean = calculate_angle(vertical_ref, keypoints[11], keypoints[5])
            
            current_wrist_y = keypoints[9][1]
            # Handle first frame of video (velocity is 0)
            if prev_wrist_y == 0: prev_wrist_y = current_wrist_y
            
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

            # --- AUTO SAVE (LABEL = 1) ---
            # We hardcode the label to 1 since folder is "good_videos"
            label = 1
            
            data_row = [label, 
                        elbow_offset, back_lean, wrist_velocity, 
                        shoulder_shrug, elbow_flare, neck_angle,
                        rep_phase] + flat_keypoints.tolist()
            
            with open(output_file, mode='a', newline='') as f:
                csv.writer(f).writerow(data_row)
            
            frame_count += 1

    cap.release()
    print(f"   -> Finished {video_file} ({frame_count} frames saved)")

cv2.destroyAllWindows()
print("--- ALL VIDEOS PROCESSED ---")