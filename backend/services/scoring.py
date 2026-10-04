"""
Transparent Prototype Review-Priority Scoring Engine.
Calculates explainable 0-100 scores based on documented rules in config/scoring_rules.json.
Provides full breakdown of contributing rules and explicit prototype heuristic disclaimer.
Neutralizes shared unverified status so scoring is truly discriminative.
Wires R_INVALID_GEOMETRY and R_EMPTY_GEOMETRY directly into scoring calculations.
Supports parcel-level score aggregation and red/amber/green (RAG) classification.
"""
import json
import os
from typing import Dict, Any, List, Optional
from backend.models.schemas import ReviewScoreBreakdown, ScoreRuleContribution

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "scoring_rules.json")

def load_scoring_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_review_score(
    feature: Dict[str, Any],
    associated_warning_ids: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    associated_warnings: Optional[List[Any]] = None,
) -> ReviewScoreBreakdown:
    """
    Computes explainable review score and returns structured breakdown.
    Marked with disclaimer: prototype heuristic—not a validated survey-priority model.
    """
    if config is None:
        config = load_scoring_config()
        
    feat_id = feature.get("id") or feature.get("properties", {}).get("feature_id", "UNKNOWN")
    props = feature.get("properties", {})
    feat_type = props.get("feature_type", "building")
    source = props.get("source", "project_vaayu_sample")
    review_status = props.get("review_status", "unverified")
    confidence = props.get("confidence")
    area_sqm = props.get("area_sqm") or 0.0
    
    current_score = config.get("base_score", 10)
    triggered_rules: List[ScoreRuleContribution] = []
    
    # 1. Unverified source rule (Score-neutral informational factor when points == 0)
    if source in ["project_vaayu_sample", "synthetic_test", "osm"]:
        rule = next((r for r in config["rules"] if r["rule_id"] == "R_UNVERIFIED_SOURCE"), None)
        if rule:
            current_score += rule["points"]
            triggered_rules.append(ScoreRuleContribution(
                rule_id=rule["rule_id"],
                rule_name=rule["rule_name"],
                description=rule["description"],
                points=rule["points"]
            ))
            
    # 2. Geometry anomaly / invalid / empty geometry
    warning_ids_list = associated_warning_ids or []
    warning_records = [
        warning.model_dump() if hasattr(warning, "model_dump") else dict(warning)
        for warning in (associated_warnings or [])
    ]
    has_invalid_geom = any("INVALID" in wid or "CORRUPT" in wid for wid in warning_ids_list)
    has_empty_geom = any("EMPTY" in wid for wid in warning_ids_list)
    
    if has_invalid_geom:
        rule = next((r for r in config["rules"] if r["rule_id"] == "R_INVALID_GEOMETRY"), None)
        if rule:
            current_score += rule["points"]
            triggered_rules.append(ScoreRuleContribution(
                rule_id=rule["rule_id"],
                rule_name=rule["rule_name"],
                description=rule["description"],
                points=rule["points"]
            ))
            
    if has_empty_geom:
        rule = next((r for r in config["rules"] if r["rule_id"] == "R_EMPTY_GEOMETRY"), None)
        if rule:
            current_score += rule["points"]
            triggered_rules.append(ScoreRuleContribution(
                rule_id=rule["rule_id"],
                rule_name=rule["rule_name"],
                description=rule["description"],
                points=rule["points"]
            ))
            
    # 3. Spatial overlap or topology warning rule (other than geometry anomalies)
    overlap_warnings = [
        wid for wid in warning_ids_list 
        if not ("INVALID" in wid or "CORRUPT" in wid or "EMPTY" in wid)
    ]
    if len(overlap_warnings) > 0:
        rule = next((r for r in config["rules"] if r["rule_id"] == "R_SPATIAL_OVERLAP_WARNING"), None)
        if rule:
            relevant_records = [
                warning for warning in warning_records
                if warning.get("warning_id") in overlap_warnings
            ]
            evidence_summary = [
                {
                    "warning_id": warning.get("warning_id"),
                    "warning_type": warning.get("warning_type"),
                    "evidence": warning.get("evidence", {}),
                    "tolerance": warning.get("tolerance", {}),
                }
                for warning in relevant_records
            ]
            measured_text = "; ".join(
                f"{warning.get('warning_id')}: " + ", ".join(
                    f"{name}={value}" for name, value in warning.get("evidence", {}).items()
                    if isinstance(value, (int, float))
                )
                for warning in relevant_records
            )
            current_score += rule["points"]
            triggered_rules.append(ScoreRuleContribution(
                rule_id=rule["rule_id"],
                rule_name=rule["rule_name"],
                description=(
                    f"{rule['description']} ({len(overlap_warnings)} warnings active: "
                    f"{', '.join(overlap_warnings[:3])})"
                    + (f" Measured evidence: {measured_text}." if measured_text else "")
                ),
                points=rule["points"],
                evidence={"warnings": evidence_summary} if evidence_summary else {},
                suggested_actions=[
                    warning["suggested_action"] for warning in relevant_records
                    if warning.get("suggested_action")
                ],
            ))
            
    # 4. Low model confidence (for AI building model)
    if source == "ai_building_model" and confidence is not None and confidence < 0.70:
        rule = next((r for r in config["rules"] if r["rule_id"] == "R_LOW_AI_CONFIDENCE"), None)
        if rule:
            current_score += rule["points"]
            triggered_rules.append(ScoreRuleContribution(
                rule_id=rule["rule_id"],
                rule_name=rule["rule_name"],
                description=f"Model confidence ({confidence:.2f}) is below 0.70 threshold.",
                points=rule["points"]
            ))
            
    # 5. Extreme footprint dimensions (tiny or massive)
    if feat_type in ["building", "synthetic_building"] and area_sqm > 0 and (area_sqm < 10.0 or area_sqm > 600.0):
        rule = next((r for r in config["rules"] if r["rule_id"] == "R_EXTREME_DIMENSIONS"), None)
        if rule:
            current_score += rule["points"]
            triggered_rules.append(ScoreRuleContribution(
                rule_id=rule["rule_id"],
                rule_name=rule["rule_name"],
                description=f"Footprint area ({area_sqm:.1f} m²) is outside typical residential building envelope (10-600 m²).",
                points=rule["points"]
            ))
            
    # 6. Human review status rules
    if review_status == "approved":
        rule = next((r for r in config["rules"] if r["rule_id"] == "R_HUMAN_APPROVED"), None)
        if rule:
            current_score += rule["points"]
            triggered_rules.append(ScoreRuleContribution(
                rule_id=rule["rule_id"],
                rule_name=rule["rule_name"],
                description=rule["description"],
                points=rule["points"]
            ))
    elif review_status == "rejected":
        rule = next((r for r in config["rules"] if r["rule_id"] == "R_HUMAN_REJECTED"), None)
        if rule:
            current_score += rule["points"]
            triggered_rules.append(ScoreRuleContribution(
                rule_id=rule["rule_id"],
                rule_name=rule["rule_name"],
                description=rule["description"],
                points=rule["points"]
            ))
            
    # Clamp score to [0, 100]
    final_score = max(0, min(100, current_score))
    
    # Determine priority tier
    if final_score >= 70:
        priority = "high"
    elif final_score >= 40:
        priority = "medium"
    else:
        priority = "low"
        
    return ReviewScoreBreakdown(
        feature_id=feat_id,
        feature_type=feat_type,
        total_score=final_score,
        priority=priority,
        heuristic_disclaimer=config.get("disclaimer", "prototype heuristic—not a validated survey-priority model"),
        rules_triggered=triggered_rules
    )


def aggregate_parcel_scores(
    parcel_feature: Dict[str, Any],
    intersecting_buildings: List[Dict[str, Any]],
    intersecting_roads: List[Dict[str, Any]],
    associated_warnings: List[str]
) -> Dict[str, Any]:
    """
    Aggregates parcel-level score and determines Red/Amber/Green (RAG) classification.
    Red: high conflict (score >= 60 or active crossing/overlap warnings or invalid geometry)
    Amber: moderate conflict (score 30-59 or unreviewed/extreme buildings)
    Green: low conflict (score < 30, no active topology conflicts)
    """
    parcel_id = parcel_feature.get("id") or parcel_feature.get("properties", {}).get("feature_id", "UNKNOWN")
    is_synthetic = "synthetic" in parcel_feature.get("properties", {}).get("source", "")
    
    # Derive indicators
    bld_count = len(intersecting_buildings)
    road_count = len(intersecting_roads)
    warning_count = len(associated_warnings)
    
    # Base parcel score
    base = 10
    score = base
    contributing_reasons = []
    
    if is_synthetic:
        contributing_reasons.append("Demonstration fixture: synthetic parcel")
        
    if warning_count > 0:
        add_pts = min(50, warning_count * 25)
        score += add_pts
        contributing_reasons.append(f"{warning_count} active topology warnings ({add_pts} pts)")
        
    if road_count > 0:
        score += 20
        contributing_reasons.append(f"Intersects {road_count} road corridor(s) (+20 pts)")
        
    if bld_count > 0:
        # Check building warnings or extreme sizes
        extreme_blds = [b for b in intersecting_buildings if (b.get("properties", {}).get("area_sqm") or 0) > 600 or (b.get("properties", {}).get("area_sqm") or 0) < 10]
        if extreme_blds:
            score += 15
            contributing_reasons.append(f"{len(extreme_blds)} building(s) with unusual dimensions (+15 pts)")
            
    final_score = max(0, min(100, score))
    
    # RAG band
    if final_score >= 60 or warning_count > 0:
        rag_status = "RED"
        rag_label = "Requires Survey Priority Review"
    elif final_score >= 30:
        rag_status = "AMBER"
        rag_label = "Secondary Visual Inspection"
    else:
        rag_status = "GREEN"
        rag_label = "Clear / Low Conflict"
        
    return {
        "parcel_id": parcel_id,
        "is_synthetic": is_synthetic,
        "synthetic_badge": "Synthetic test data — not real parcels" if is_synthetic else None,
        "rag_status": rag_status,
        "rag_label": rag_label,
        "parcel_score": final_score,
        "intersecting_building_count": bld_count,
        "intersecting_road_count": road_count,
        "active_warning_ids": associated_warnings,
        "contributing_reasons": contributing_reasons,
        "heuristic_disclaimer": "prototype heuristic—not a validated survey-priority model"
    }
