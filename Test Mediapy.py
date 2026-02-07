import mediapipe as mp

print("--- DEBUG INFO ---")
print(f"Mediapipe version: {mp.__version__}")
print(f"Does it have solutions? {'solutions' in dir(mp)}")