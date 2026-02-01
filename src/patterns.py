import time
from collections import deque

# a deviation must hold for this many consecutive frames to count
SUSTAIN_FRAMES = 30  # ~1 second at 30fps

# z-score threshold: below this, a deviation is noise
DEVIATION_FLOOR = 1.5

# flat detection: all deltas must stay within this band
FLAT_CEILING = 0.4

# jaw rigidity: z-score at or below this indicates clenching (near baseline or compressed)
JAW_RIGID_CEILING = 0.3

# blink rate rolling window
BLINK_WINDOW = 5.0

# per-pattern confirmation: pattern conditions must hold for this many
# consecutive detect() calls before the pattern is emitted
CONFIRM_CALLS = 2

PATTERN_RULES = [
    {
        "name": "blink_spike",
        "description": "Elevated blink frequency",
        "requires": {"blink_rate": "high"},
        "signal_count": 1,
    },
    {
        "name": "blink_suppression",
        "description": "Reduced blink frequency",
        "requires": {"blink_rate": "low"},
        "signal_count": 1,
    },
    {
        "name": "eye_narrowing",
        "description": "Sustained reduction in eye openness",
        "requires": {"eye_openness": "low"},
        "signal_count": 1,
    },
    {
        "name": "jaw_tension",
        "description": "Sustained jaw rigidity",
        "requires": {"jaw_displacement": "rigid"},
        "signal_count": 1,
    },
    {
        "name": "mouth_clamp",
        "description": "Sustained mouth compression",
        "requires": {"mouth_openness": "low"},
        "signal_count": 1,
    },
    {
        "name": "facial_constraint",
        "description": "Jaw rigidity with mouth compression",
        "requires": {"jaw_displacement": "rigid", "mouth_openness": "low"},
        "signal_count": 2,
    },
    {
        "name": "facial_freeze",
        "description": "Near-total cessation of movement across signals",
        "requires": {"eye_openness": "flat", "mouth_openness": "flat", "jaw_displacement": "flat"},
        "signal_count": 3,
    },
    {
        "name": "cognitive_load",
        "description": "Blink spike with jaw tension",
        "requires": {"blink_rate": "high", "jaw_displacement": "rigid"},
        "signal_count": 2,
    },
    {
        "name": "suppression_cluster",
        "description": "Eye narrowing with mouth compression",
        "requires": {"eye_openness": "low", "mouth_openness": "low"},
        "signal_count": 2,
    },
    {
        "name": "arousal_spike",
        "description": "Widened eyes with jaw drop",
        "requires": {"eye_openness": "high", "mouth_openness": "high"},
        "signal_count": 2,
    },
    {
        "name": "composure_performance",
        "description": "Flat signals despite high-pressure context",
        "requires": {"eye_openness": "flat", "jaw_displacement": "flat"},
        "signal_count": 2,
    },
]


class PatternDetector:
    def __init__(self, fps=30):
        self.fps = fps
        self.sustain_frames = max(1, int(SUSTAIN_FRAMES * (fps / 30.0)))

        # per-signal: deque of consecutive delta values (no timestamps needed)
        self.delta_buffer = {}

        # per-signal: count of consecutive frames meeting threshold
        self.streak_high = {}
        self.streak_low = {}
        self.streak_flat = {}
        self.streak_rigid = {}

        # blink timestamps
        self.blink_times = deque()
        self.baseline_blink_rate = 0.0

        # per-pattern confirmation counter
        self.pattern_confirm = {}

    def set_baseline_blink_rate(self, rate):
        self.baseline_blink_rate = rate

    def feed(self, deltas, signals, timestamp):
        for key, val in deltas.items():
            if key == "blink_frame":
                continue

            # update high streak
            if val > DEVIATION_FLOOR:
                self.streak_high[key] = self.streak_high.get(key, 0) + 1
            else:
                self.streak_high[key] = 0

            # update low streak
            if val < -DEVIATION_FLOOR:
                self.streak_low[key] = self.streak_low.get(key, 0) + 1
            else:
                self.streak_low[key] = 0

            # update flat streak
            if abs(val) < FLAT_CEILING:
                self.streak_flat[key] = self.streak_flat.get(key, 0) + 1
            else:
                self.streak_flat[key] = 0

            # update rigid streak (near zero or negative — no expansion)
            if val <= JAW_RIGID_CEILING:
                self.streak_rigid[key] = self.streak_rigid.get(key, 0) + 1
            else:
                self.streak_rigid[key] = 0

        # track blinks
        if signals.get("blink_frame", 0.0) == 1.0:
            self.blink_times.append(timestamp)

        cutoff = timestamp - BLINK_WINDOW
        while self.blink_times and self.blink_times[0] < cutoff:
            self.blink_times.popleft()

    def _check_signal(self, signal_name, direction):
        if signal_name == "blink_rate":
            return self._check_blink_rate(direction)

        if direction == "high":
            return self.streak_high.get(signal_name, 0) >= self.sustain_frames
        elif direction == "low":
            return self.streak_low.get(signal_name, 0) >= self.sustain_frames
        elif direction == "flat":
            return self.streak_flat.get(signal_name, 0) >= self.sustain_frames
        elif direction == "rigid":
            return self.streak_rigid.get(signal_name, 0) >= self.sustain_frames
        return False

    def _check_blink_rate(self, direction):
        current_rate = len(self.blink_times) / BLINK_WINDOW
        baseline = self.baseline_blink_rate

        if baseline < 0.01:
            if direction == "high" and current_rate > 0.6:
                return True
            if direction == "low":
                return current_rate == 0 and len(self.blink_times) == 0
            return False

        ratio = current_rate / baseline
        if direction == "high":
            return ratio > 2.0
        elif direction == "low":
            return ratio < 0.3
        return False

    def detect(self):
        detected = []

        for rule in PATTERN_RULES:
            all_met = True
            for signal, direction in rule["requires"].items():
                if not self._check_signal(signal, direction):
                    all_met = False
                    break

            name = rule["name"]

            if all_met:
                self.pattern_confirm[name] = self.pattern_confirm.get(name, 0) + 1
            else:
                self.pattern_confirm[name] = 0

            # only emit after sustained confirmation
            if self.pattern_confirm.get(name, 0) >= CONFIRM_CALLS:
                confidence = "medium" if rule["signal_count"] >= 2 else "low"
                detected.append({
                    "pattern": name,
                    "description": rule["description"],
                    "confidence": confidence,
                })

        # return empty list if nothing confirmed — no null padding
        return detected

    def reset(self):
        self.streak_high.clear()
        self.streak_low.clear()
        self.streak_flat.clear()
        self.streak_rigid.clear()
        self.blink_times.clear()
        self.pattern_confirm.clear()


def format_patterns(patterns):
    if not patterns:
        return ""
    lines = []
    for p in patterns:
        lines.append(f"  [{p['confidence'].upper():6s}] {p['pattern']:25s} — {p['description']}")
    return "\n".join(lines)
