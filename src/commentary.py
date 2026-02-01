import os
import json
import random

BANK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "commentary_bank.json")

_bank = None
_used = {}  # tracks used indices per pattern+confidence to avoid repeats within a session


def _load_bank():
    global _bank
    if _bank is None:
        with open(BANK_PATH, "r") as f:
            _bank = json.load(f)
    return _bank


def reset_session():
    global _used
    _used = {}


def get_commentary(pattern_name, confidence):
    bank = _load_bank()

    if pattern_name not in bank:
        return None

    entries = bank[pattern_name].get(confidence)
    if not entries:
        entries = bank[pattern_name].get("low", [])
    if not entries:
        return None

    # track used indices to avoid repeating within a session
    key = f"{pattern_name}:{confidence}"
    if key not in _used:
        _used[key] = []

    available = [i for i in range(len(entries)) if i not in _used[key]]
    if not available:
        # all used, reset for this key
        _used[key] = []
        available = list(range(len(entries)))

    idx = random.choice(available)
    _used[key].append(idx)

    entry = entries[idx]
    return {
        "observation": entry["observation"],
        "interpretation": entry["interpretation"],
    }


_NULL_TEXT = "No sustained facial patterns were detected during the response window."
_MAX_OBSERVATIONS = 3


def generate_question_commentary(aggregated_patterns):
    if not aggregated_patterns:
        return [{
            "observation": _NULL_TEXT,
            "interpretation": "",
        }]

    # aggregated_patterns arrives sorted by hits (descending) from aggregate_patterns()
    # take only the top patterns by sustained duration / hit count
    top = aggregated_patterns[:_MAX_OBSERVATIONS]

    results = []
    for p in top:
        c = get_commentary(p["pattern"], p["confidence"])
        if c:
            results.append(c)

    if not results:
        return [{
            "observation": _NULL_TEXT,
            "interpretation": "",
        }]

    return results


def format_commentary(commentary_list):
    lines = []
    for c in commentary_list:
        lines.append(f"  {c['observation']}")
        if c["interpretation"]:
            lines.append(f"  {c['interpretation']}")
        lines.append("")
    return "\n".join(lines).rstrip()
