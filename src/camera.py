import os
import time
import threading
import cv2
import numpy as np
import mediapipe as mp
from signals import extract_signals, compute_baseline, compute_deltas
from patterns import PatternDetector, format_patterns

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

WINDOW_NAME = "NO ALIBI"
TARGET_FPS = 30
CALIBRATION_DURATION = 10
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# shared state for async callback
latest_landmarks = None
landmarks_lock = threading.Lock()


def on_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_landmarks
    with landmarks_lock:
        if result.face_landmarks:
            latest_landmarks = result.face_landmarks[0]
        else:
            latest_landmarks = None


def draw_landmarks(frame, face_landmarks):
    h, w, _ = frame.shape
    points = []
    for lm in face_landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

    # draw connections for face oval, eyes, lips
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]

    for indices, color in [(FACE_OVAL, (100, 100, 100)), (LEFT_EYE, (0, 200, 200)),
                           (RIGHT_EYE, (0, 200, 200)), (LIPS_OUTER, (0, 100, 200))]:
        for i in range(len(indices) - 1):
            if indices[i] < len(points) and indices[i + 1] < len(points):
                cv2.line(frame, points[indices[i]], points[indices[i + 1]], color, 1)


# init camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera not found.")
    exit(1)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=on_result,
)

landmarker = FaceLandmarker.create_from_options(options)

print("NO ALIBI — Camera active. Press 'q' to exit.")

# state
calibrating = True
calibration_start = None
signal_history = []
baseline = None
blink_count = 0
last_blink_state = False
frame_count = 0
detector = PatternDetector(fps=TARGET_FPS)
last_pattern_print = 0
PATTERN_PRINT_INTERVAL = 3.0  # seconds between pattern reports
active_patterns = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    output = frame.copy()
    h, w, _ = output.shape

    with landmarks_lock:
        current_landmarks = latest_landmarks

    if current_landmarks is not None:
        draw_landmarks(output, current_landmarks)
        signals = extract_signals(current_landmarks)

        is_blink = signals["blink_frame"] == 1.0
        if is_blink and not last_blink_state:
            blink_count += 1
        last_blink_state = is_blink

        if calibrating:
            if calibration_start is None:
                calibration_start = time.time()
                print("CALIBRATION STARTED — Look at the screen. Sit still.")

            elapsed = time.time() - calibration_start
            remaining = max(0, CALIBRATION_DURATION - elapsed)
            signal_history.append(signals)

            cv2.putText(output, "CALIBRATING", (20, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(output, f"Hold still... {remaining:.1f}s", (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

            bar_w = int((elapsed / CALIBRATION_DURATION) * (w - 40))
            cv2.rectangle(output, (20, h - 90), (20 + min(bar_w, w - 40), h - 75), (0, 200, 255), -1)
            cv2.rectangle(output, (20, h - 90), (w - 20, h - 75), (0, 200, 255), 1)

            if elapsed >= CALIBRATION_DURATION:
                calibrating = False
                calibration_elapsed = time.time() - calibration_start
                blinks_per_sec = blink_count / calibration_elapsed if calibration_elapsed > 0 else 0.0
                baseline = compute_baseline(signal_history)
                baseline["blink_rate"] = {
                    "mean": blinks_per_sec,
                    "std": max(blinks_per_sec * 0.3, 0.1),
                }
                print("\n--- BASELINE ESTABLISHED ---")
                for key, val in baseline.items():
                    print(f"  {key:20s}  mean={val['mean']:.4f}  std={val['std']:.4f}")
                print(f"  {'blinks_total':20s}  {blink_count} in {calibration_elapsed:.1f}s")
                print("----------------------------\n")
                blink_count = 0
                last_blink_state = False
                detector.set_baseline_blink_rate(baseline["blink_rate"]["mean"])

        else:
            now = time.time()
            deltas = compute_deltas(signals, baseline)
            detector.feed(deltas, signals, now)

            frame_count += 1
            if frame_count % 15 == 0:
                delta_str = "  ".join(f"{k}={v:+.2f}s" for k, v in deltas.items() if k != "blink_frame")
                print(f"LIVE | {delta_str}")

            # run pattern detection periodically
            if now - last_pattern_print >= PATTERN_PRINT_INTERVAL:
                active_patterns = detector.detect()
                if active_patterns:
                    print(f"\n--- PATTERNS DETECTED ---")
                    print(format_patterns(active_patterns))
                    print(f"-------------------------\n")
                last_pattern_print = now

            # on-screen delta display
            y_offset = 80
            for key, val in deltas.items():
                if key == "blink_frame":
                    continue
                color = (0, 255, 0) if abs(val) < 1.5 else (0, 165, 255) if abs(val) < 2.5 else (0, 0, 255)
                label = f"{key}: {val:+.2f} sigma"
                cv2.putText(output, label, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                y_offset += 25

            # on-screen pattern display
            y_pattern = y_offset + 15
            for p in active_patterns:
                conf_color = (0, 200, 255) if p["confidence"] == "medium" else (180, 180, 180)
                cv2.putText(output, f"{p['pattern']} [{p['confidence']}]", (20, y_pattern),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1, cv2.LINE_AA)
                y_pattern += 22

            cv2.putText(output, "LIVE — Observing", (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(output, "NO FACE DETECTED", (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(output, "NO ALIBI", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, output)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
