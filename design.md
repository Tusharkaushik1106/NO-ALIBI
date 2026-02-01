# NO ALIBI – Design Document

## Overview
NO ALIBI is a webcam-driven interrogation experience where users answer situational questions while their facial reactions are observed and reflected back in a detective-style narrative.

The system performs:
- Real-time facial signal detection
- Rule-based interpretation
- Narrative feedback generation

No claims of truth detection are made.

---

## Core Flow
1. Camera initializes
2. Interrogation question appears
3. Timed response window
4. Facial signals recorded during response
5. Observations generated
6. Interpretations inferred
7. Detective-style feedback displayed
8. Final personality file compiled

---

## Design Principles
- Experience > accuracy
- Observation > judgment
- Uncertainty is a feature
- Brutal honesty without accusation

---

## System Layers

### 1. Detection Layer
- Webcam input
- Face landmarks
- Temporal signal tracking

### 2. Interpretation Layer
- Rule-based mappings
- Confidence scoring (low/medium only)
- No single-signal conclusions

### 3. Narrative Layer
- Detective commentary engine
- Personality file generator
- Replayable outcomes

---

## Explicit Non-Goals
- No emotion prediction models
- No psychological scoring
- No user advice or coaching

---

## Technical Architecture

### Component List

| Component | Responsibility | Technology |
|---|---|---|
| **Camera Module** | Webcam access, frame capture, resolution normalization | Browser MediaDevices API |
| **Face Tracker** | Landmark detection (468 points), head pose estimation | MediaPipe Face Mesh (pre-trained, no training required) |
| **Signal Extractor** | Derives discrete signals from raw landmarks per frame | Custom JS/TS — pure math on landmark coordinates |
| **Baseline Engine** | Establishes per-subject neutral behavior during calibration phase | Rolling averages over first 30–60s of neutral conversation |
| **Deviation Detector** | Compares live signals against subject's own baseline | Threshold-based rules with configurable sensitivity |
| **Signal Aggregator** | Clusters co-occurring deviations into named patterns | Rule table: if signals A + B within window T → pattern P |
| **Confidence Scorer** | Assigns low/medium confidence to each pattern | Fixed rules: single signal = low, cluster = medium, never high |
| **Commentary Engine** | Selects and populates observational commentary templates | Template bank with conditional slot-filling |
| **Profile Compiler** | Aggregates patterns across all questions into final subject profile | Accumulator with weighted pattern frequency |
| **UI Controller** | Manages question flow, timing, camera overlay, narrative display | Frontend framework (React or vanilla DOM) |

---

### Data Flow

```
Step 1: CAPTURE
  Browser MediaDevices API → raw video frames (30fps target)
  No frames are stored. Processing is real-time, in-memory only.

Step 2: TRACK
  Raw frame → MediaPipe Face Mesh → 468 facial landmarks + head rotation
  Output: landmark coordinate array per frame

Step 3: EXTRACT
  Landmarks → Signal Extractor computes:
    - Eye aspect ratio (blink detection)
    - Gaze vector (eye contact direction)
    - Mouth aperture + lip compression ratio
    - Jaw displacement
    - Brow position (corrugator/frontalis region)
    - Head pitch/yaw/roll deltas
    - Nostril width delta
  Output: signal vector per frame

Step 4: BASELINE
  During calibration phase (pre-interrogation neutral prompt):
    Signal vectors → rolling mean + standard deviation per signal
  Output: subject-specific baseline profile

Step 5: DETECT
  During interrogation:
    Live signal vector vs. baseline → deviation = (current - mean) / stddev
    If deviation exceeds threshold → flag signal as active
  Output: list of active deviations per time window

Step 6: AGGREGATE
  Active deviations within a time window → pattern matcher
  Rule examples:
    - blink_rate_increase + jaw_tension → "cognitive load cluster"
    - gaze_break_down + lip_compression → "suppression cluster"
    - head_stillness + blink_reduction → "freeze pattern"
  Output: named pattern(s) with confidence level

Step 7: INTERPRET
  Named pattern + confidence level → Commentary Engine
  Engine selects from pre-written template pairs:
    - Observation sentence (what was seen)
    - Interpretation sentence (what it may suggest)
  Output: commentary text for display

Step 8: NARRATE
  Commentary rendered in UI with timed reveal (typewriter cadence)
  Displayed after subject finishes responding to each question

Step 9: COMPILE
  After all 12 questions:
    Accumulated patterns → Profile Compiler
    Fills personality profile template sections:
      Cognitive Style / Stress Response / Moral Friction /
      Self-Presentation / Behavioral Summary
  Output: final subject profile file
```

---

### Uncertainty Handling

| Location | Mechanism |
|---|---|
| **Signal Extractor** | Signals below a minimum deviation threshold are discarded, not downgraded. No noise enters the pipeline. |
| **Deviation Detector** | All thresholds are calibrated per-subject. No universal cutoffs. A tense person's baseline tension is not flagged as tension. |
| **Confidence Scorer** | Hard ceiling at medium. The system structurally cannot produce high-confidence outputs. Single-signal detections are always low. |
| **Commentary Engine** | All interpretation sentences use hedged language: "suggests," "likely," "pattern observed." No template contains certainty language. |
| **Profile Compiler** | Profile sections use bracket notation for variable fills, preserving the conditional nature of every observation. No flattening into definitive statements. |
| **Fallback: No Signal** | If insufficient deviations are detected for a question, the system outputs a null-observation commentary (e.g., "Nothing registered. That in itself is a data point.") rather than fabricating a reading. |

---

### UI Disclaimer Placement

| Screen | Disclaimer |
|---|---|
| **Launch / Landing** | "This is a narrative experience. It does not detect lies, diagnose conditions, or assess mental health. It observes patterns and reflects them back. Nothing more." |
| **Camera Permission** | "Video is processed locally in real-time. No frames are stored, transmitted, or recorded." |
| **Calibration Start** | "Establishing your baseline. There is no correct way to look." |
| **During Interrogation** | No disclaimers. Immersion is not interrupted. The landing disclaimer covers the session. |
| **Profile Reveal** | Small footer: "This profile is generated from observable behavioral patterns during a simulated interrogation. It is not a psychological assessment." |
| **Share / Export** | "This file contains observational notes from a narrative game. It does not constitute any form of evaluation or diagnosis." |

---

## 7-Day MVP Plan

> Assumption: Solo developer. Python + basic frontend. Experience over polish.

---

### Day 1 — Camera and Face Tracking Pipeline

**Objective:** Get a working webcam feed with real-time facial landmark overlay.

**Features built:**
- Browser webcam capture via MediaDevices API
- MediaPipe Face Mesh integration (468 landmarks rendering on canvas)
- Frame rate stabilization (target 30fps, graceful fallback to 15fps)
- Basic HTML page with video element and canvas overlay

**Deliverable:** A page that opens the camera and draws facial landmarks on the user's face in real-time. No processing. Just proof that tracking works.

---

### Day 2 — Signal Extraction and Baseline Engine

**Objective:** Convert raw landmarks into named signals and establish per-subject baselines.

**Features built:**
- Signal Extractor: compute eye aspect ratio, gaze vector, mouth aperture, lip compression, jaw displacement, brow position, head pose deltas from landmark coordinates
- Signal vector output per frame (structured object)
- Baseline Engine: 30-second calibration phase with neutral prompt ("Look at the screen. Relax. We're just getting to know your face.")
- Rolling mean + standard deviation computed per signal during calibration
- Console output showing live signal values vs. baseline

**Deliverable:** Calibration runs, baseline is stored in memory, and live deviations from baseline print to console during free interaction.

---

### Day 3 — Deviation Detection and Pattern Aggregation

**Objective:** Detect meaningful deviations and cluster them into named patterns.

**Features built:**
- Deviation Detector: z-score comparison (current vs. baseline), threshold-based flagging
- Minimum deviation filter (discard noise below threshold)
- Signal Aggregator: rule table mapping co-occurring signals to named patterns
  - Implement at least 6 pattern rules (cognitive load, suppression, freeze, arousal, composure performance, null)
- Confidence Scorer: single signal = low, cluster = medium, ceiling enforced
- Time-windowed aggregation (per-question observation window)

**Deliverable:** Ask a test question verbally, observe the console output naming detected patterns with confidence levels. The system sees something and labels it.

---

### Day 4 — Interrogation Flow and Question Engine

**Objective:** Build the question delivery system and timed response capture.

**Features built:**
- Question bank: all 12 interrogation questions stored as structured data (text, category, pressure type)
- Sequential question delivery with timed display
- Response window: fixed duration per question (15–20 seconds) with visual countdown
- Signal recording scoped to each question's active window
- Transition screens between questions (brief black screen or minimal text)
- Landing screen with disclaimer
- Camera permission screen with privacy notice

**Deliverable:** Full 12-question interrogation flow runs end to end. Camera tracks. Signals extract. Patterns detect per question. No narrative output yet — just structured data per question logged to console.

---

### Day 5 — Commentary Engine and Narrative Display

**Objective:** Turn detected patterns into detective-style observational text and display it.

**Features built:**
- Commentary template bank: 20+ observation/interpretation pairs mapped to pattern types
- Template selection logic: pattern name + confidence level → appropriate commentary pair
- Null-observation fallback commentary for questions with insufficient signal
- Narrative display UI: typewriter-effect text reveal after each question
- Dark, minimal UI theme (interrogation aesthetic — dark background, monospace font, muted palette)
- Commentary appears between questions as a "detective reviews the subject" interstitial

**Deliverable:** Complete interrogation with live narrative feedback after each question. The system observes the subject, names a pattern, and delivers a commentary pair on screen.

---

### Day 6 — Profile Compiler and Results Screen

**Objective:** Aggregate all observations into a final personality profile and present it.

**Features built:**
- Profile Compiler: accumulate patterns across all 12 questions
- Frequency analysis: which patterns appeared most, which questions triggered strongest deviations
- Template fill for all 5 profile sections (Cognitive Style, Stress Response, Moral Friction, Self-Presentation, Behavioral Summary)
- Bracket notation selection logic (choose the appropriate fill from each template based on accumulated data)
- Results screen: profile displayed section by section with timed reveals
- Disclaimer footer on results screen
- Export: generate downloadable plain-text file of the profile

**Deliverable:** After the final question, the system compiles and displays a complete subject profile. Subject can download it as a text file.

---

### Day 7 — Polish, Edge Cases, and Playtest

**Objective:** Harden the experience and run it end to end multiple times.

**Features built:**
- Edge case handling: face lost mid-question (pause detection, resume), poor lighting fallback messaging
- Calibration retry if baseline is too noisy (auto-detect high variance, prompt re-calibration)
- Null-signal graceful handling across entire session (if subject shows minimal deviation overall, profile reflects that explicitly rather than guessing)
- UI transitions smoothed (fade between states, no jarring cuts)
- Audio consideration: optional ambient low drone or silence (silence is default — audio is a stretch goal)
- Full end-to-end playtest: run 3–5 complete sessions, note where the experience breaks or feels wrong
- Bug fixes from playtest

**Deliverable:** A working MVP that a person can sit down in front of, answer 12 questions, receive real-time observational commentary, and walk away with a downloaded personality profile. Rough edges acceptable. The experience must be coherent from start to finish.

---

### MVP Success Criteria

The MVP is complete when:
- [ ] Camera initializes and tracks without manual intervention
- [ ] Baseline calibration completes in under 60 seconds
- [ ] All 12 questions are delivered with timed response windows
- [ ] At least 1 commentary pair is generated per question (even if null-observation)
- [ ] Final profile is generated with all 5 sections populated
- [ ] Profile can be exported as a text file
- [ ] No certainty language appears anywhere in the output
- [ ] Disclaimers are present at launch and on results screen