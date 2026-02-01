import os
import json
import random
from datetime import datetime

CASE_FILE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# minimum number of questions a pattern must appear in to be considered recurring
MIN_QUESTION_RECURRENCE = 2

# pattern-to-section mapping
COGNITIVE_PATTERNS = {"blink_spike", "blink_suppression", "cognitive_load"}
STRESS_PATTERNS = {"facial_freeze", "composure_performance", "arousal_spike"}
CONTROL_PATTERNS = {"jaw_tension", "mouth_clamp", "facial_constraint", "suppression_cluster", "eye_narrowing"}


def _aggregate_session(session_log):
    """Build cross-question pattern frequency map.

    Returns dict: pattern_name -> {
        question_count: int (number of distinct questions it appeared in),
        questions: list of question_ids,
        tags: list of tags for those questions,
        total_hits: int,
    }
    """
    cross = {}
    for entry in session_log:
        qid = entry["question_id"]
        tag = entry.get("tag", "")
        seen_this_q = set()
        for p in entry.get("patterns", []):
            name = p["pattern"]
            if name not in cross:
                cross[name] = {
                    "question_count": 0,
                    "questions": [],
                    "tags": [],
                    "total_hits": 0,
                }
            if name not in seen_this_q:
                cross[name]["question_count"] += 1
                cross[name]["questions"].append(qid)
                cross[name]["tags"].append(tag)
                seen_this_q.add(name)
            cross[name]["total_hits"] += p.get("hits", 1)
    return cross


def _recurring_patterns(cross):
    """Filter to patterns appearing in >= MIN_QUESTION_RECURRENCE questions."""
    return {
        name: info for name, info in cross.items()
        if info["question_count"] >= MIN_QUESTION_RECURRENCE
    }


def _patterns_in_set(recurring, pattern_set):
    """Return list of (name, info) from recurring that belong to pattern_set, sorted by question_count desc."""
    found = [(n, i) for n, i in recurring.items() if n in pattern_set]
    found.sort(key=lambda x: -x[1]["question_count"])
    return found


def _section_overview(session_log, recurring):
    lines = []
    total_q = len(session_log)
    q_with_patterns = sum(1 for e in session_log if e.get("pattern_count", 0) > 0)
    recurring_count = len(recurring)

    lines.append("OVERVIEW")
    lines.append("-" * 40)

    if q_with_patterns == 0:
        lines.append("No sustained facial patterns were detected across the session.")
        lines.append("The subject maintained a baseline-consistent presentation throughout.")
        return "\n".join(lines)

    lines.append(f"Across {total_q} questions, sustained facial patterns were observed in {q_with_patterns}.")

    if recurring_count > 0:
        lines.append(f"Of those, {recurring_count} pattern type{'s' if recurring_count != 1 else ''} recurred across multiple questions.")
    else:
        lines.append("No pattern type recurred across multiple questions.")
        lines.append("Observed signals were isolated and do not form a cross-session tendency.")

    return "\n".join(lines)


def _section_cognitive(recurring):
    lines = []
    lines.append("COGNITIVE STYLE")
    lines.append("-" * 40)

    found = _patterns_in_set(recurring, COGNITIVE_PATTERNS)
    if not found:
        lines.append("No recurring cognitive-load indicators were observed across the session.")
        return "\n".join(lines)

    for name, info in found:
        q_count = info["question_count"]
        if name == "blink_spike":
            lines.append(
                f"A recurring elevation in blink rate above baseline was observed across {q_count} questions. "
                "This may indicate a tendency toward elevated processing demand when formulating responses."
            )
        elif name == "blink_suppression":
            lines.append(
                f"Blink suppression below baseline recurred across {q_count} questions. "
                "This may indicate a pattern of attentional narrowing during response windows."
            )
        elif name == "cognitive_load":
            lines.append(
                f"Co-occurring blink elevation and jaw rigidity were observed across {q_count} questions. "
                "This may indicate sustained processing demand paired with physical containment."
            )

    # tag context
    all_tags = set()
    for _, info in found:
        all_tags.update(info["tags"])
    all_tags.discard("")
    if all_tags:
        lines.append(f"These patterns were concentrated in questions tagged: {', '.join(sorted(all_tags))}.")

    return "\n".join(lines)


def _section_stress(recurring):
    lines = []
    lines.append("STRESS REGULATION")
    lines.append("-" * 40)

    found = _patterns_in_set(recurring, STRESS_PATTERNS)
    if not found:
        lines.append("No recurring stress-regulation patterns were observed across the session.")
        return "\n".join(lines)

    for name, info in found:
        q_count = info["question_count"]
        if name == "facial_freeze":
            lines.append(
                f"Near-total cessation of facial movement was observed across {q_count} questions. "
                "This may indicate a recurring freeze-adjacent response under questioning pressure."
            )
        elif name == "composure_performance":
            lines.append(
                f"Unusually narrow signal variance near baseline was observed across {q_count} questions. "
                "This may indicate active composure maintenance. Whether natural or performed is not determinable."
            )
        elif name == "arousal_spike":
            lines.append(
                f"Simultaneous above-baseline elevation of eye and mouth aperture recurred across {q_count} questions. "
                "This may indicate a pattern of unregulated initial response prior to correction."
            )

    all_tags = set()
    for _, info in found:
        all_tags.update(info["tags"])
    all_tags.discard("")
    if all_tags:
        lines.append(f"These patterns surfaced during questions tagged: {', '.join(sorted(all_tags))}.")

    return "\n".join(lines)


def _section_control(recurring):
    lines = []
    lines.append("RESPONSE CONTROL")
    lines.append("-" * 40)

    found = _patterns_in_set(recurring, CONTROL_PATTERNS)
    if not found:
        lines.append("No recurring response-control patterns were observed across the session.")
        return "\n".join(lines)

    for name, info in found:
        q_count = info["question_count"]
        if name == "jaw_tension":
            lines.append(
                f"Jaw rigidity below baseline movement range recurred across {q_count} questions. "
                "This may indicate a tendency toward physical containment during response formulation."
            )
        elif name == "mouth_clamp":
            lines.append(
                f"Mouth compression below baseline recurred across {q_count} questions. "
                "This may indicate a pattern of suppressing initial verbal responses before speaking."
            )
        elif name == "facial_constraint":
            lines.append(
                f"Co-occurring jaw rigidity and mouth compression were observed across {q_count} questions. "
                "This may indicate coordinated physical restraint of facial output."
            )
        elif name == "suppression_cluster":
            lines.append(
                f"Simultaneous reduction in eye openness and mouth aperture recurred across {q_count} questions. "
                "This may indicate multi-region facial containment as a recurring response pattern."
            )
        elif name == "eye_narrowing":
            lines.append(
                f"Sustained reduction in eye aperture below baseline recurred across {q_count} questions. "
                "This may indicate a tendency toward evaluative narrowing during questioning."
            )

    all_tags = set()
    for _, info in found:
        all_tags.update(info["tags"])
    all_tags.discard("")
    if all_tags:
        lines.append(f"These patterns were observed during questions tagged: {', '.join(sorted(all_tags))}.")

    return "\n".join(lines)


def _section_behavioral_summary(session_log, recurring):
    lines = []
    lines.append("BEHAVIORAL SUMMARY")
    lines.append("-" * 40)

    if not recurring:
        lines.append(
            "The subject did not exhibit recurring facial patterns across the session. "
            "Observed signals were either absent or isolated to individual questions. "
            "No cross-session behavioral tendency can be identified from this data."
        )
        return "\n".join(lines)

    # determine dominant section
    cog = _patterns_in_set(recurring, COGNITIVE_PATTERNS)
    stress = _patterns_in_set(recurring, STRESS_PATTERNS)
    control = _patterns_in_set(recurring, CONTROL_PATTERNS)

    cog_weight = sum(i["question_count"] for _, i in cog)
    stress_weight = sum(i["question_count"] for _, i in stress)
    control_weight = sum(i["question_count"] for _, i in control)

    total_q = len(session_log)
    q_with = sum(1 for e in session_log if e.get("pattern_count", 0) > 0)

    if q_with <= total_q * 0.25:
        lines.append(
            "Across the session, observable patterns were sparse. "
            "The subject presented with minimal sustained deviation from baseline. "
            "This absence itself is a data point, though its meaning is not determinable from observation alone."
        )
    else:
        segments = []
        if cog_weight > 0:
            segments.append("processing-load indicators")
        if stress_weight > 0:
            segments.append("stress-regulation signals")
        if control_weight > 0:
            segments.append("response-control behaviors")

        if segments:
            joined = ", ".join(segments[:-1])
            if len(segments) > 1:
                joined += f", and {segments[-1]}"
            else:
                joined = segments[0]
            lines.append(
                f"The session produced recurring {joined}. "
                "Taken together, these patterns suggest a subject who engages in measurable "
                "facial adjustment during morally loaded questioning."
            )

        # strongest tendency
        dominant = max(
            [("cognitive-load", cog_weight), ("stress-regulation", stress_weight), ("response-control", control_weight)],
            key=lambda x: x[1],
        )
        if dominant[1] > 0:
            lines.append(
                f"The most recurrent tendency fell in the {dominant[0]} category. "
                "This does not imply a cause or a character trait. It is a pattern, observed and recorded."
            )

    return "\n".join(lines)


def _section_closing():
    lines = []
    lines.append("CLOSING NOTE")
    lines.append("-" * 40)
    lines.append(
        "This case file was generated from observable facial patterns during a simulated interrogation. "
        "It is not a psychological assessment, a lie detection report, or a diagnostic instrument."
    )
    lines.append("")
    lines.append(
        "All observations are relative to an individual baseline established at the start of the session. "
        "The system does not know what the subject was thinking, feeling, or intending. "
        "It records deviations. It does not explain them."
    )
    lines.append("")
    lines.append(
        "Patterns that appeared in isolation were excluded. "
        "Patterns that recurred were noted. "
        "Recurrence does not establish meaning."
    )
    return "\n".join(lines)


def generate_case_file(session_log):
    """Generate the full case file from session data.

    Returns the case file as a string.
    """
    cross = _aggregate_session(session_log)
    recurring = _recurring_patterns(cross)

    sections = []

    # header
    header_lines = [
        "=" * 50,
        "  NO ALIBI — CASE FILE",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 50,
        "",
    ]
    sections.append("\n".join(header_lines))

    sections.append(_section_overview(session_log, recurring))
    sections.append("")
    sections.append(_section_cognitive(recurring))
    sections.append("")
    sections.append(_section_stress(recurring))
    sections.append("")
    sections.append(_section_control(recurring))
    sections.append("")
    sections.append(_section_behavioral_summary(session_log, recurring))
    sections.append("")
    sections.append(_section_closing())

    # footer
    sections.append("")
    sections.append("=" * 50)
    sections.append("  END OF CASE FILE")
    sections.append("=" * 50)

    return "\n".join(sections)


def save_case_file(session_log):
    """Generate and save the case file. Returns the file path."""
    text = generate_case_file(session_log)

    os.makedirs(CASE_FILE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"case_file_{timestamp}.txt"
    path = os.path.join(CASE_FILE_DIR, filename)

    with open(path, "w") as f:
        f.write(text)

    return path, text
