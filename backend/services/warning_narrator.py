"""Deterministic plain-language summaries generated from warning payloads."""
from __future__ import annotations

from typing import Any, Dict, List


def _format_value(value: Any, unit: str = "") -> str:
    if isinstance(value, float):
        rendered = f"{value:.3f}".rstrip("0").rstrip(".")
    else:
        rendered = str(value)
    return f"{rendered} {unit}".strip()


def narration_status() -> Dict[str, Any]:
    return {"enabled": False, "model": None, "mode": "deterministic"}


def _deterministic_text(item: Dict[str, Any]) -> str:
    evidence = item.get("evidence") or {}
    tolerance = item.get("tolerance") or {}
    action = item.get("suggested_action") or {}
    if not evidence:
        return str(item.get("explanation", ""))

    feature_ids = item.get("feature_ids") or []
    feature_text = f" for feature(s) {', '.join(map(str, feature_ids))}" if feature_ids else ""
    warning_type = str(item.get("warning_type", "review rule")).replace("_", " ")
    details = []
    unit = str(tolerance.get("unit", ""))
    for name, value in evidence.items():
        if name in {"geometry_type", "validity_reason", "parse_error", "nearest_parcel_id"}:
            continue
        label = name.removesuffix("_m2").removesuffix("_m").replace("_", " ")
        details.append(f"{label}: {_format_value(value, unit if name.endswith(('_m', '_m2')) else '')}")

    sentence = f"{warning_type.capitalize()}{feature_text}."
    if details:
        sentence += f" Measured evidence: {'; '.join(details)}."
    if evidence.get("validity_reason"):
        sentence += f" Validity detail: {evidence['validity_reason']}."
    if evidence.get("parse_error"):
        sentence += f" Parse detail: {evidence['parse_error']}."
    if tolerance.get("name") and "value" in tolerance:
        sentence += (
            f" Review tolerance: {tolerance['name'].replace('_', ' ')} is "
            f"{_format_value(tolerance['value'], unit)}."
        )
    if action.get("text"):
        sentence += f" Suggested action: {str(action['text']).rstrip('.!?')}."
    return sentence


def narrate_warnings(warnings: List[Any]) -> Dict[str, Any]:
    """Return deterministic summaries based exclusively on supplied warning fields."""
    selected = list(warnings or [])[:20]
    narratives: Dict[str, Dict[str, str]] = {}
    for warning in selected:
        item = warning.model_dump() if hasattr(warning, "model_dump") else dict(warning)
        warning_id = str(item.get("warning_id", ""))
        narratives[warning_id] = {"text": _deterministic_text(item), "source": "template"}
    return {
        "enabled": False,
        "model": None,
        "mode": "deterministic",
        "narratives": narratives,
        "fallback": True,
    }
