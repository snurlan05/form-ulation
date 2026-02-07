import mediapipe
import os

print("--- DEBUG INFO ---")
print(f"I am loading mediapipe from: {mediapipe.__file__}")
print(f"Does it have solutions? {'solutions' in dir(mediapipe)}")