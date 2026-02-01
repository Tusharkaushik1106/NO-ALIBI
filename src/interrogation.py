import os
import sys
import json
import time
import threading
import cv2
import numpy as np
import mediapipe as mp
from signals import extract_signals, compute_baseline, compute_deltas
from patterns import PatternDetector, format_patterns
from commentary import generate_question_commentary, format_commentary, reset_session as reset_commentary

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

WINDOW_NAME = "NO ALIBI"
TARGET_FPS = 30
CALIBRATION_DURATION = 10
QUESTION_DURATION = 15
TRANSITION_DURATION = 3
COMMENTARY_DURATION = 8  # seconds to display commentary
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "questions.json")

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


def wrap_text(text, max_chars=50):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines


def load_questions():
    with open(QUESTIONS_PATH, "r") as f:
        return json.load(f)


# minimum number of sample hits for a pattern to count as sustained
MIN_SUSTAINED_HITS = 2  # at 1s sample interval over 15s window, 2 hits = >=2s sustained

PATTERN_SAMPLE_INTERVAL = 1.0  # sample every 1 second for finer granularity


def aggregate_patterns(raw_events):
    """Deduplicate and duration-validate raw pattern events for a single question.

    Returns a list of unique patterns that appeared in at least MIN_SUSTAINED_HITS
    samples, with the highest confidence observed.
    """
    if not raw_events:
        return []

    # group by pattern name: count hits, track max confidence
    groups = {}
    for event in raw_events:
        name = event["pattern"]
        conf = event["confidence"]
        if name not in groups:
            groups[name] = {"hits": 0, "confidence": conf, "first_seen": event["timestamp"]}
        groups[name]["hits"] += 1
        if conf == "medium":
            groups[name]["confidence"] = "medium"

    # filter: only patterns sustained across enough samples
    result = []
    for name, info in groups.items():
        if info["hits"] >= MIN_SUSTAINED_HITS:
            result.append({
                "pattern": name,
                "confidence": info["confidence"],
                "hits": info["hits"],
            })

    # sort by hits descending, then confidence (medium first)
    result.sort(key=lambda p: (-p["hits"], 0 if p["confidence"] == "medium" else 1))
    return result


# --- phases ---

PHASE_DISCLAIMER = "disclaimer"
PHASE_CALIBRATION = "calibration"
PHASE_QUESTION = "question"
PHASE_COMMENTARY = "commentary"
PHASE_TRANSITION = "transition"
PHASE_COMPLETE = "complete"


def main():
    global latest_landmarks

    questions = load_questions()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera not found.")
        sys.exit(1)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    detector = PatternDetector(fps=TARGET_FPS)

    # state
    phase = PHASE_DISCLAIMER
    phase_start = time.time()

    # calibration state
    signal_history = []
    baseline = None
    blink_count = 0
    last_blink_state = False

    # interrogation state
    current_q_index = 0
    question_patterns = []  # collected patterns per question window
    session_log = []  # final log: list of dicts per question

    # pattern sampling during questions
    last_pattern_sample = 0

    # commentary state
    current_commentary = []
    commentary_text_flat = ""  # full text for typewriter
    commentary_chars_shown = 0

    reset_commentary()

    print("=" * 50)
    print("  NO ALIBI — Interrogation Session")
    print("=" * 50)

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
        now = time.time()

        with landmarks_lock:
            current_landmarks = latest_landmarks

        # --- PHASE: DISCLAIMER ---
        if phase == PHASE_DISCLAIMER:
            overlay = np.zeros_like(output)
            lines = [
                "NO ALIBI",
                "",
                "This is a narrative experience.",
                "It does not detect lies, diagnose conditions,",
                "or assess mental health.",
                "",
                "It observes patterns and reflects them back.",
                "Nothing more.",
                "",
                "Press SPACE to begin.",
            ]
            y = h // 2 - len(lines) * 15
            for line in lines:
                if line == "NO ALIBI":
                    cv2.putText(overlay, line, (w // 2 - 120, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(overlay, line, (w // 2 - 250, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
                y += 35
            output = overlay

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                phase = PHASE_CALIBRATION
                phase_start = time.time()
                print("\n[PHASE] Calibration started.")
            elif key == ord("q"):
                break

            cv2.imshow(WINDOW_NAME, output)
            continue

        # --- PHASE: CALIBRATION ---
        if phase == PHASE_CALIBRATION:
            if current_landmarks is not None:
                draw_landmarks(output, current_landmarks)
                signals = extract_signals(current_landmarks)
                signal_history.append(signals)

                is_blink = signals["blink_frame"] == 1.0
                if is_blink and not last_blink_state:
                    blink_count += 1
                last_blink_state = is_blink

            elapsed = now - phase_start
            remaining = max(0, CALIBRATION_DURATION - elapsed)

            cv2.putText(output, "CALIBRATING", (20, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(output, f"Look at the screen. Sit still. {remaining:.1f}s", (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

            bar_w = int((elapsed / CALIBRATION_DURATION) * (w - 40))
            cv2.rectangle(output, (20, h - 90), (20 + min(bar_w, w - 40), h - 75), (0, 200, 255), -1)
            cv2.rectangle(output, (20, h - 90), (w - 20, h - 75), (0, 200, 255), 1)

            if elapsed >= CALIBRATION_DURATION and len(signal_history) > 0:
                cal_elapsed = now - phase_start
                blinks_per_sec = blink_count / cal_elapsed if cal_elapsed > 0 else 0.0
                baseline = compute_baseline(signal_history)
                baseline["blink_rate"] = {
                    "mean": blinks_per_sec,
                    "std": max(blinks_per_sec * 0.3, 0.1),
                }
                detector.set_baseline_blink_rate(baseline["blink_rate"]["mean"])

                print("\n--- BASELINE ESTABLISHED ---")
                for key, val in baseline.items():
                    print(f"  {key:20s}  mean={val['mean']:.4f}  std={val['std']:.4f}")
                print(f"  {'blinks_total':20s}  {blink_count} in {cal_elapsed:.1f}s")
                print("----------------------------\n")

                blink_count = 0
                last_blink_state = False
                phase = PHASE_QUESTION
                phase_start = now
                detector.reset()
                question_patterns = []
                last_pattern_sample = now
                q = questions[current_q_index]
                print(f"\n[Q{q['id']}] {q['text']}")

        # --- PHASE: QUESTION ---
        elif phase == PHASE_QUESTION:
            q = questions[current_q_index]
            elapsed = now - phase_start
            remaining = max(0, QUESTION_DURATION - elapsed)

            if current_landmarks is not None:
                draw_landmarks(output, current_landmarks)
                signals = extract_signals(current_landmarks)
                deltas = compute_deltas(signals, baseline)
                detector.feed(deltas, signals, now)

                # sample patterns periodically
                if now - last_pattern_sample >= PATTERN_SAMPLE_INTERVAL:
                    detected = detector.detect()
                    if detected:
                        for p in detected:
                            question_patterns.append({
                                "pattern": p["pattern"],
                                "confidence": p["confidence"],
                                "timestamp": round(elapsed, 2),
                            })
                    last_pattern_sample = now

            # draw question text
            text_lines = wrap_text(q["text"], max_chars=55)
            y_text = 80
            for line in text_lines:
                cv2.putText(output, line, (20, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                y_text += 28

            # question number and timer
            cv2.putText(output, f"{q['id']} / {len(questions)}", (w - 160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

            # countdown bar
            bar_frac = remaining / QUESTION_DURATION
            bar_color = (0, 255, 0) if remaining > 5 else (0, 165, 255) if remaining > 2 else (0, 0, 255)
            bar_w = int(bar_frac * (w - 40))
            cv2.rectangle(output, (20, h - 40), (20 + bar_w, h - 25), bar_color, -1)
            cv2.rectangle(output, (20, h - 40), (w - 20, h - 25), (100, 100, 100), 1)
            cv2.putText(output, f"{remaining:.1f}s", (w - 80, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1, cv2.LINE_AA)

            if elapsed >= QUESTION_DURATION:
                # final pattern sample
                detected = detector.detect()
                if detected:
                    for p in detected:
                        question_patterns.append({
                            "pattern": p["pattern"],
                            "confidence": p["confidence"],
                            "timestamp": round(elapsed, 2),
                        })

                # aggregate: deduplicate and duration-validate
                aggregated = aggregate_patterns(question_patterns)

                # log this question
                entry = {
                    "question_id": q["id"],
                    "tag": q.get("tag", ""),
                    "raw_events": len(question_patterns),
                    "patterns": aggregated,
                    "pattern_count": len(aggregated),
                }
                session_log.append(entry)

                # generate commentary
                current_commentary = generate_question_commentary(aggregated)

                # print to console
                print(f"\n--- Q{q['id']} OBSERVATION ---")
                print(format_commentary(current_commentary))
                print(f"------------------------")

                # store commentary in log
                entry["commentary"] = current_commentary

                # prepare typewriter display
                lines_for_display = []
                for c in current_commentary:
                    lines_for_display.append(c["observation"])
                    if c["interpretation"]:
                        lines_for_display.append(c["interpretation"])
                    lines_for_display.append("")
                commentary_text_flat = "\n".join(lines_for_display).rstrip()
                commentary_chars_shown = 0

                # move to commentary phase
                detector.reset()
                question_patterns = []
                phase = PHASE_COMMENTARY
                phase_start = now

        # --- PHASE: COMMENTARY ---
        elif phase == PHASE_COMMENTARY:
            elapsed = now - phase_start
            overlay = np.zeros_like(output)

            # typewriter: reveal chars over first 4 seconds
            typewriter_duration = min(4.0, COMMENTARY_DURATION * 0.5)
            if typewriter_duration > 0 and len(commentary_text_flat) > 0:
                frac = min(1.0, elapsed / typewriter_duration)
                commentary_chars_shown = int(frac * len(commentary_text_flat))
            else:
                commentary_chars_shown = len(commentary_text_flat)

            visible_text = commentary_text_flat[:commentary_chars_shown]
            display_lines = visible_text.split("\n")

            # header
            q = questions[current_q_index]
            cv2.putText(overlay, f"Q{q['id']} — OBSERVED", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1, cv2.LINE_AA)

            y_line = 100
            for line in display_lines:
                wrapped = wrap_text(line, max_chars=65)
                for wl in wrapped:
                    cv2.putText(overlay, wl, (30, y_line),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                    y_line += 24
                if not wrapped:
                    y_line += 12

            output = overlay

            if elapsed >= COMMENTARY_DURATION:
                current_q_index += 1
                if current_q_index >= len(questions):
                    phase = PHASE_COMPLETE
                    phase_start = now
                else:
                    phase = PHASE_TRANSITION
                    phase_start = now

        # --- PHASE: TRANSITION ---
        elif phase == PHASE_TRANSITION:
            elapsed = now - phase_start
            overlay = np.zeros_like(output)

            q_next = questions[current_q_index]
            cv2.putText(overlay, f"Question {q_next['id']} / {len(questions)}", (w // 2 - 130, h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(overlay, f"{q_next.get('tag', '')}", (w // 2 - 50, h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
            output = overlay

            # keep feeding landmarks during transition for continuity
            if current_landmarks is not None:
                signals = extract_signals(current_landmarks)
                deltas = compute_deltas(signals, baseline)
                detector.feed(deltas, signals, now)

            if elapsed >= TRANSITION_DURATION:
                phase = PHASE_QUESTION
                phase_start = now
                last_pattern_sample = now
                q = questions[current_q_index]
                print(f"\n[Q{q['id']}] {q['text']}")

        # --- PHASE: COMPLETE ---
        elif phase == PHASE_COMPLETE:
            overlay = np.zeros_like(output)
            cv2.putText(overlay, "INTERROGATION COMPLETE", (w // 2 - 220, h // 2 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, "Session data logged. Press 'q' to exit.", (w // 2 - 220, h // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
            output = overlay

            # print full session log once
            if phase_start != 0:
                print("\n" + "=" * 50)
                print("  SESSION LOG")
                print("=" * 50)
                for entry in session_log:
                    qid = entry["question_id"]
                    if entry["patterns"]:
                        patterns_str = ", ".join(
                            f"{p['pattern']}[{p['confidence']}]"
                            for p in entry["patterns"]
                        )
                    else:
                        patterns_str = "--"
                    tag = entry.get("tag", "")
                    print(f"  {qid} [{tag:12s}]  {patterns_str}")
                q_with = sum(1 for e in session_log if e["pattern_count"] > 0)
                print(f"\n  Questions with sustained patterns: {q_with} / {len(session_log)}")
                print("=" * 50)
                phase_start = 0  # prevent re-printing

        # title
        if phase not in (PHASE_DISCLAIMER,):
            cv2.putText(output, "NO ALIBI", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # save session log to file
    log_path = os.path.join(os.path.dirname(__file__), "..", "data", "session_log.json")
    with open(log_path, "w") as f:
        json.dump(session_log, f, indent=2)
    print(f"\nSession log saved to {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
