"""Shared pose / feature-extraction logic for Form-ulation.

This module is a direct, factored-out copy of the inference logic that
already existed in ``App.py`` (Streamlit) and ``live_demo.py`` (OpenCV).
It is deliberately *not* a new model: the same YOLOv8n-pose keypoints,
the same 7 engineered biomechanical features, in the same order, are fed
to the same ``bicep_model_v2.pkl`` Random Forest bundle.

Keypoint indices are YOLOv8 / COCO pose indices (``keypoints.xyn``,
normalised x/y):
    3  = left ear
    5  = left shoulder
    7  = left elbow
    9  = left wrist
    11 = left hip

Note the quirk inherited from the original code: ``elbow_offset`` and
``elbow_flare`` are computed identically (``|kp[7].x - kp[5].x|``).  The
model was trained with that duplication, so both columns are kept in
exactly the same positions — do not "fix" it without retraining.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Feature order expected by bicep_model_v2.pkl (verified by loading the
# bundle: features == [7 engineered, K0..K33]).
ENGINEERED_FEATURES = [
    "ELBOW_OFFSET",
    "BACK_LEAN",
    "WRIST_VELOCITY",
    "SHOULDER_SHRUG",
    "ELBOW_FLARE",
    "NECK_ANGLE",
    "REP_PHASE",
]


def calculate_angle(a, b, c):
    """Angle ABC in degrees. Identical to the helper in App.py."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


class FrameState:
    """Per-video state carried across frames (same as App.py's class)."""

    def __init__(self):
        self.prev_wrist_y = 0.0
        self.prev_elbow_angle = 180.0


def extract_frame_features(keypoints, state: FrameState):
    """Build one model input row from a (17, 2) normalised keypoint array.

    Returns ``(input_values, diagnostics)`` where ``input_values`` is the
    41-element list in the model's feature order, and ``diagnostics`` holds
    the human-readable per-frame values (elbow angle, feature values).
    """
    elbow_offset = abs(keypoints[7][0] - keypoints[5][0])
    vertical_ref = [keypoints[11][0], keypoints[11][1] - 0.5]
    back_lean = calculate_angle(vertical_ref, keypoints[11], keypoints[5])

    current_wrist_y = keypoints[9][1]
    wrist_velocity = (
        0
        if state.prev_wrist_y == 0
        else abs(current_wrist_y - state.prev_wrist_y) * 100
    )
    state.prev_wrist_y = current_wrist_y

    shoulder_shrug = abs(keypoints[3][1] - keypoints[5][1])
    elbow_flare = abs(keypoints[7][0] - keypoints[5][0])
    neck_angle = calculate_angle(keypoints[3], keypoints[5], keypoints[11])

    current_elbow_angle = calculate_angle(keypoints[5], keypoints[7], keypoints[9])
    angle_change = current_elbow_angle - state.prev_elbow_angle
    rep_phase = 1 if angle_change < -1 else (-1 if angle_change > 1 else 0)
    state.prev_elbow_angle = current_elbow_angle

    flat_keypoints = keypoints.flatten()
    input_values = [
        elbow_offset,
        back_lean,
        wrist_velocity,
        shoulder_shrug,
        elbow_flare,
        neck_angle,
        rep_phase,
    ] + flat_keypoints.tolist()

    diagnostics = {
        "elbow_offset": float(elbow_offset),
        "back_lean": float(back_lean),
        "wrist_velocity": float(wrist_velocity),
        "shoulder_shrug": float(shoulder_shrug),
        "elbow_flare": float(elbow_flare),
        "neck_angle": float(neck_angle),
        "rep_phase": int(rep_phase),
        "elbow_angle": float(current_elbow_angle),
    }
    return input_values, diagnostics


def predict_frame(model, scaler, feature_cols, input_values):
    """Run one frame through the scaler + Random Forest. Returns (pred, prob)."""
    input_df = pd.DataFrame([input_values], columns=feature_cols)
    input_scaled = scaler.transform(input_df)
    prediction = int(model.predict(input_scaled)[0])
    probability = float(model.predict_proba(input_scaled)[0][prediction])
    return prediction, probability


def estimate_reps(elbow_angles):
    """Heuristic rep count from the elbow-angle sequence.

    A rep is counted when the arm goes from extended (>150 deg) to flexed
    (<90 deg) and back to extended. This is an estimate only — the
    original live demos did not count reps, they classified per frame.
    """
    reps = 0
    flexed = False
    for angle in elbow_angles:
        if not flexed and angle < 90:
            flexed = True
        elif flexed and angle > 150:
            reps += 1
            flexed = False
    return reps
