"""
Turns raw patient vitals into explicit, human-readable flags against standard
adult reference ranges — e.g. "Heart rate 132 bpm is ABOVE the normal range
(60-100 bpm): tachycardia."

Why this exists rather than just dumping `patient_context.vitals` as JSON
into the prompt: an LLM asked to eyeball a JSON blob and *also* infer what's
abnormal is doing two error-prone things at once (reading structured data +
clinical interpretation). Pre-computing the flag deterministically removes
one of those from the LLM's job entirely — the interpretation of "132 bpm is
high" is a lookup, not a generation, so it shouldn't be delegated to a model
that can hallucinate.

Reference ranges are intentionally coarse, adult, non-context-specific
values for demo purposes — not a substitute for age/condition-adjusted
clinical reference ranges in a real deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VitalStatus(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"
    UNKNOWN_VITAL = "unknown_vital"


@dataclass
class VitalFlag:
    name: str
    value: float
    status: VitalStatus
    reference_range: tuple[float, float] | None
    note: str

    def as_evidence_line(self) -> str:
        if self.status == VitalStatus.NORMAL:
            return f"{self.name}: {self.value} (within normal range {self.reference_range})."
        if self.status == VitalStatus.UNKNOWN_VITAL:
            return f"{self.name}: {self.value} (no reference range configured — not automatically flagged)."
        direction = "ABOVE" if self.status == VitalStatus.HIGH else "BELOW"
        return f"{self.name}: {self.value} is {direction} the normal range {self.reference_range} — {self.note}"


# (low, high, label-if-high, label-if-low)
_REFERENCE_RANGES: dict[str, tuple[float, float, str, str]] = {
    "heart_rate": (60.0, 100.0, "tachycardia", "bradycardia"),
    "respiratory_rate": (12.0, 20.0, "tachypnea", "bradypnea"),
    "temperature_c": (36.1, 37.2, "fever", "hypothermia"),
    "spo2": (95.0, 100.0, "hypoxemia", "hypoxemia"),
    "systolic_bp": (90.0, 120.0, "hypertension", "hypotension"),
    "diastolic_bp": (60.0, 80.0, "hypertension", "hypotension"),
    "blood_glucose_mg_dl": (70.0, 140.0, "hyperglycemia", "hypoglycemia"),
    "heart_rate_bpm": (60.0, 100.0, "tachycardia", "bradycardia"),
}


def analyze_vitals(vitals: dict[str, float]) -> list[VitalFlag]:
    """
    Flags each vital against a known reference range. Unrecognized vital
    names are returned with status=UNKNOWN_VITAL rather than dropped or
    guessed at — silently ignoring an unrecognized vital would be exactly
    the kind of quiet data loss this module exists to avoid.
    """
    flags: list[VitalFlag] = []
    for raw_name, value in (vitals or {}).items():
        key = raw_name.strip().lower().replace(" ", "_")
        ref = _REFERENCE_RANGES.get(key)

        if ref is None:
            flags.append(VitalFlag(name=raw_name, value=value, status=VitalStatus.UNKNOWN_VITAL, reference_range=None, note=""))
            continue

        low, high, high_label, low_label = ref
        if value > high:
            flags.append(VitalFlag(raw_name, value, VitalStatus.HIGH, (low, high), high_label))
        elif value < low:
            flags.append(VitalFlag(raw_name, value, VitalStatus.LOW, (low, high), low_label))
        else:
            flags.append(VitalFlag(raw_name, value, VitalStatus.NORMAL, (low, high), ""))
    return flags


def vitals_to_evidence_text(vitals: dict[str, float]) -> str:
    """Renders flagged vitals as newline-separated evidence lines for the synthesis prompt."""
    flags = analyze_vitals(vitals)
    if not flags:
        return ""
    return "\n".join(f.as_evidence_line() for f in flags)
