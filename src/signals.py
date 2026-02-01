import math
import numpy as np

# MediaPipe Face Mesh landmark indices
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

UPPER_LIP = 13
LOWER_LIP = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

CHIN = 152
NOSE_BRIDGE = 6


def _dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def extract_signals(face_landmarks):
    """Extract signals from a list of NormalizedLandmark objects."""
    lm = face_landmarks

    left_ear = _dist(lm[LEFT_EYE_TOP], lm[LEFT_EYE_BOTTOM]) / max(_dist(lm[LEFT_EYE_INNER], lm[LEFT_EYE_OUTER]), 1e-6)
    right_ear = _dist(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM]) / max(_dist(lm[RIGHT_EYE_INNER], lm[RIGHT_EYE_OUTER]), 1e-6)
    eye_openness = (left_ear + right_ear) / 2.0

    mouth_vertical = _dist(lm[UPPER_LIP], lm[LOWER_LIP])
    mouth_horizontal = max(_dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT]), 1e-6)
    mouth_openness = mouth_vertical / mouth_horizontal

    jaw_displacement = _dist(lm[CHIN], lm[NOSE_BRIDGE])

    blink_frame = 1.0 if eye_openness < 0.15 else 0.0

    return {
        "eye_openness": eye_openness,
        "mouth_openness": mouth_openness,
        "jaw_displacement": jaw_displacement,
        "blink_frame": blink_frame,
    }


def compute_baseline(signal_history):
    baseline = {}
    for key in signal_history[0]:
        values = np.array([s[key] for s in signal_history])
        baseline[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return baseline


def compute_deltas(signals, baseline):
    deltas = {}
    for key, value in signals.items():
        mean = baseline[key]["mean"]
        std = baseline[key]["std"]
        if std < 1e-6:
            deltas[key] = 0.0
        else:
            deltas[key] = (value - mean) / std
    return deltas
