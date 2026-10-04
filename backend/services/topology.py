"""
Deterministic Topology Validation Engine using Shapely and CRS-aware metric geometry.
Detects:
1. Invalid / empty / self-intersecting geometry (safe handling via geometry_utils)
2. Overlapping polygons within a layer (using metric EPSG:32643 intersection area)
3. Building-road spatial intersections (neutral wording: 'spatial overlap—review visually')
4. Building-parcel crossing / straddling (evaluated strictly on non-empty parcel layer)
5. Enforces strictly: NO real parcel conflict warnings when real parcel layer has 0 features;
   reports 'not evaluated: no parcel boundaries loaded'.
"""
from typing import Dict, Any, List, Optional
import shapely.geometry
from shapely.validation import explain_validity
from backend.models.schemas import TopologyWarning
from backend.services.geometry_utils import (
    calculate_metric_area_sqm,
    calculate_metric_distance_meters,
    calculate_metric_intersection,
    project_to_metric,
)
from backend.services.review_config import (
    LOW_CONFIDENCE_THRESHOLD,
    MIN_CROSSING_AREA_M2,
    MIN_CROSSING_DEPTH_M,
    MIN_OVERLAP_AREA_M2,
    SEVERITY_RATIO_CUTOFFS,
    SUGGESTED_ACTION_LOOKUP,
    TOLERANCE_LOOKUP,
)

NEUTRAL_ROAD_OVERLAP_EXPLANATION = (
    "Spatial overlap detected between building {bld_id} and road corridor {road_id} (approx {area:.1f} m²)—review visually. "
    "Road corridors and centerlines are spatial references, not legal rights-of-way."
)

SYNTHETIC_ROAD_OVERLAP_EXPLANATION = (
    "Demonstration Warning: Spatial overlap detected between synthetic building {bld_id} "
    "and synthetic road corridor {road_id} (approx {area:.1f} m²)—review visually."
)

SYNTHETIC_PARCEL_CROSSING_EXPLANATION = (
    "Demonstration Warning: Synthetic building footprint {bld_id} crosses boundary of synthetic parcel {parcel_id}. "
    "This is a demonstration fixture; real Lalpur parcel dataset is 0 features."
)

REAL_PARCEL_CROSSING_EXPLANATION = (
    "Building footprint {bld_id} crosses parcel boundary {parcel_id}. "
    "Review boundaries visually against imagery."
)

SYNTHETIC_OVERLAP_EXPLANATION = (
    "Demonstration Warning: Synthetic polygons {id1} and {id2} overlap spatially by approximately {area:.1f} m². "
    "Requires human vertex reconciliation."
)

REAL_OVERLAP_EXPLANATION = (
    "Spatial overlap detected between polygon {id1} and polygon {id2} (approx {area:.1f} m²). "
    "Review boundaries visually."
)


def _severity_for_ratio(ratio: float) -> str:
    if ratio >= SEVERITY_RATIO_CUTOFFS["high"]:
        return "high"
    if ratio >= SEVERITY_RATIO_CUTOFFS["medium"]:
        return "medium"
    return "low"


def _review_fields(
    warning_type: str,
    measured_name: str,
    measured_value: float,
    *,
    evidence: Optional[Dict[str, Any]] = None,
    lower_is_worse: bool = False,
    detection_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    tolerance = TOLERANCE_LOOKUP.get(warning_type, {})
    tolerance_value = float(tolerance.get("value", 0.0))
    if lower_is_worse:
        ratio = tolerance_value / max(measured_value, 1e-9)
        exceedance = max(0.0, tolerance_value - measured_value)
    else:
        ratio = measured_value / tolerance_value if tolerance_value > 0 else 1.0
        exceedance = max(0.0, measured_value - tolerance_value)
    action = SUGGESTED_ACTION_LOOKUP.get(warning_type, {
        "action_type": "desk_review",
        "text": "Review the measured evidence against the source imagery and records",
    })
    return {
        "rule": f"R_{warning_type.upper()}",
        "evidence": {measured_name: measured_value, **(evidence or {})},
        "tolerance": dict(tolerance),
        "exceeds_tolerance_by": {measured_name: exceedance, "ratio": ratio},
        "detection_confidence": detection_confidence,
        "suggested_action": dict(action),
        "severity": _severity_for_ratio(ratio),
    }

def safe_parse_geometry(geom_dict: Optional[Dict[str, Any]]) -> Optional[shapely.geometry.base.BaseGeometry]:
    """Safely converts GeoJSON geometry dict to Shapely shape without throwing unhandled exceptions."""
    if not geom_dict:
        return None
    coords = geom_dict.get("coordinates")
    if coords is None or len(coords) == 0:
        return None
    try:
        shape = shapely.geometry.shape(geom_dict)
        return shape
    except Exception:
        return None

def validate_feature_geometries(features: List[Dict[str, Any]], layer_source: str = "project_vaayu_sample") -> List[TopologyWarning]:
    """Checks individual features for empty, null, or invalid geometry."""
    warnings: List[TopologyWarning] = []
    
    for feat in features:
        feat_id = feat.get("id") or feat.get("properties", {}).get("feature_id", "UNKNOWN")
        geom = feat.get("geometry")
        source = feat.get("properties", {}).get("source", layer_source)
        
        # Check empty or None geometry
        if not geom or not geom.get("coordinates") or len(geom.get("coordinates")) == 0:
            warnings.append(TopologyWarning(
                warning_id=f"W-EMPTY-{feat_id}",
                warning_type="empty_geometry",
                severity="high",
                feature_ids=[feat_id],
                explanation=f"Feature {feat_id} contains an empty or null coordinate array.",
                plain_language_rule="Features must contain valid polygon or line coordinates.",
                source=source,
                status="open",
                rule="R_EMPTY_GEOMETRY",
                evidence={"has_coordinates": False, "geometry_type": geom.get("type") if geom else None},
                tolerance={"name": "has_coordinates", "value": True, "unit": "boolean"},
                exceeds_tolerance_by={"missing_coordinates": 1.0},
                suggested_action=SUGGESTED_ACTION_LOOKUP["empty_geometry"],
            ))
            continue
            
        try:
            shape = shapely.geometry.shape(geom)
            if not shape.is_valid:
                validity_reason = explain_validity(shape)
                evidence = {"is_valid": False, "validity_reason": validity_reason}
                warnings.append(TopologyWarning(
                    warning_id=f"W-INVALID-{feat_id}",
                    warning_type="invalid_geometry",
                    severity="critical" if "synthetic" not in source else "high",
                    feature_ids=[feat_id],
                    explanation=f"Geometry for {feat_id} violates OGC standards: {validity_reason}.",
                    plain_language_rule="Non-simple or self-intersecting polygon boundary detected. Ring must not cross itself.",
                    source=source,
                    status="open",
                    affected_coordinates=[shape.centroid.x, shape.centroid.y] if not shape.is_empty else None,
                    rule="R_INVALID_GEOMETRY",
                    evidence=evidence,
                    tolerance={"name": "is_valid", "value": True, "unit": "boolean"},
                    exceeds_tolerance_by={"invalid_geometry": 1.0},
                    suggested_action=SUGGESTED_ACTION_LOOKUP["invalid_geometry"],
                ))
        except Exception as e:
            warnings.append(TopologyWarning(
                warning_id=f"W-CORRUPT-{feat_id}",
                warning_type="invalid_geometry",
                severity="critical",
                feature_ids=[feat_id],
                explanation=f"Geometry parsing error for {feat_id}: {str(e)}",
                plain_language_rule="Malformed GeoJSON geometry structure.",
                source=source,
                status="open",
                rule="R_INVALID_GEOMETRY",
                evidence={"is_valid": False, "parse_error": str(e)},
                tolerance={"name": "is_valid", "value": True, "unit": "boolean"},
                exceeds_tolerance_by={"invalid_geometry": 1.0},
                suggested_action=SUGGESTED_ACTION_LOOKUP["invalid_geometry"],
            ))
            
    return warnings

def check_polygon_overlaps(features: List[Dict[str, Any]], layer_source: str = "project_vaayu_sample") -> List[TopologyWarning]:
    """Detects mutual overlaps among polygon features in the same layer using projected metric area."""
    warnings: List[TopologyWarning] = []
    parsed_shapes = []
    
    for feat in features:
        feat_id = feat.get("id") or feat.get("properties", {}).get("feature_id", "UNKNOWN")
        source = feat.get("properties", {}).get("source", layer_source)
        shape = safe_parse_geometry(feat.get("geometry"))
        if shape and shape.is_valid and shape.geom_type in ["Polygon", "MultiPolygon"]:
            parsed_shapes.append((feat_id, source, shape))
            
    n = len(parsed_shapes)
    for i in range(n):
        id1, src1, s1 = parsed_shapes[i]
        for j in range(i + 1, n):
            id2, src2, s2 = parsed_shapes[j]
            try:
                if s1.intersects(s2):
                    inter, area_m2 = calculate_metric_intersection(s1, s2)
                    if area_m2 >= MIN_OVERLAP_AREA_M2:
                        is_synth = "synthetic" in src1 or "synthetic" in src2
                        review = _review_fields(
                            "overlapping_polygons", "overlap_area_m2", area_m2,
                            evidence={"intersection_area_m2": area_m2},
                        )
                        explanation = (
                            SYNTHETIC_OVERLAP_EXPLANATION.format(id1=id1, id2=id2, area=area_m2)
                            if is_synth else
                            REAL_OVERLAP_EXPLANATION.format(id1=id1, id2=id2, area=area_m2)
                        )
                        warnings.append(TopologyWarning(
                            warning_id=f"W-OVERLAP-{id1}-{id2}",
                            warning_type="overlapping_polygons",
                            feature_ids=[id1, id2],
                            explanation=explanation,
                            plain_language_rule="Building footprints or parcels should not overlap without a multi-level partition.",
                            source=src1 if is_synth else "project_vaayu_sample",
                            status="open",
                            affected_coordinates=[inter.centroid.x, inter.centroid.y] if inter and not inter.is_empty else None,
                            **review,
                        ))
            except Exception:
                continue
                
    return warnings

def check_building_road_intersections(
    building_features: List[Dict[str, Any]],
    road_features: List[Dict[str, Any]]
) -> List[TopologyWarning]:
    """
    Checks spatial overlaps between building polygons and road corridor polygons.
    Uses strictly NEUTRAL wording: 'spatial overlap—review visually'.
    Calculates metric area using projected UTM 43N coordinates.
    """
    warnings: List[TopologyWarning] = []
    
    buildings = []
    for b in building_features:
        bid = b.get("id") or b.get("properties", {}).get("feature_id")
        src = b.get("properties", {}).get("source", "project_vaayu_sample")
        s = safe_parse_geometry(b.get("geometry"))
        if s and s.is_valid:
            buildings.append((bid, src, s))
            
    roads = []
    for r in road_features:
        rid = r.get("id") or r.get("properties", {}).get("feature_id") or r.get("properties", {}).get("road_id")
        src = r.get("properties", {}).get("source", "project_vaayu_sample")
        s = safe_parse_geometry(r.get("geometry"))
        if s and s.is_valid:
            roads.append((rid, src, s))
            
    for bid, b_src, b_shape in buildings:
        for rid, r_src, r_shape in roads:
            try:
                if b_shape.intersects(r_shape):
                    inter, inter_area_m2 = calculate_metric_intersection(b_shape, r_shape)
                    
                    is_synth = "synthetic" in b_src or "synthetic" in r_src
                    review = _review_fields(
                        "building_road_spatial_overlap", "overlap_area_m2", inter_area_m2,
                        evidence={"intersection_area_m2": inter_area_m2},
                    )
                    explanation = (
                        SYNTHETIC_ROAD_OVERLAP_EXPLANATION.format(bld_id=bid, road_id=rid, area=inter_area_m2)
                        if is_synth else
                        NEUTRAL_ROAD_OVERLAP_EXPLANATION.format(bld_id=bid, road_id=rid, area=inter_area_m2)
                    )
                    warnings.append(TopologyWarning(
                        warning_id=f"W-RD-OVERLAP-{bid}-{rid}",
                        warning_type="building_road_spatial_overlap",
                        feature_ids=[bid, rid],
                        explanation=explanation,
                        plain_language_rule="Building footprint spatially intersects road corridor. Review imagery visually to confirm setback or alignment.",
                        source=b_src if is_synth else "project_vaayu_sample",
                        status="open",
                        affected_coordinates=[inter.centroid.x, inter.centroid.y] if inter and not inter.is_empty else None,
                        **review,
                    ))
            except Exception:
                continue
                
    return warnings

def check_building_parcel_crossings(
    buildings: List[Dict[str, Any]],
    parcels: List[Dict[str, Any]],
    is_synthetic: bool = False
) -> List[TopologyWarning]:
    """
    Checks if building footprint crosses/straddles parcel boundary.
    Strictly executed only when the provided parcel list is non-empty.
    """
    warnings: List[TopologyWarning] = []
    if not parcels or len(parcels) == 0:
        return warnings

    parsed_buildings = []
    for b in buildings:
        bid = b.get("id") or b.get("properties", {}).get("feature_id")
        bsrc = b.get("properties", {}).get("source", "synthetic_test" if is_synthetic else "project_vaayu_sample")
        bshape = safe_parse_geometry(b.get("geometry"))
        if bshape and bshape.is_valid:
            parsed_buildings.append((bid, bsrc, bshape))

    parsed_parcels = []
    for p in parcels:
        pid = p.get("id") or p.get("properties", {}).get("feature_id")
        psrc = p.get("properties", {}).get("source", "synthetic_test" if is_synthetic else "manual_visual_reference")
        pshape = safe_parse_geometry(p.get("geometry"))
        if pshape and pshape.is_valid:
            parsed_parcels.append((pid, psrc, pshape))

    for bid, bsrc, bshape in parsed_buildings:
        for pid, psrc, pshape in parsed_parcels:
            try:
                # Straddling check: intersects parcel, but is NOT completely contained within it
                if bshape.intersects(pshape) and not pshape.contains(bshape):
                    inter = bshape.intersection(pshape)
                    warning_type = "building_crosses_synthetic_parcel" if is_synthetic else "building_crosses_parcel"
                    outside_area_m2 = calculate_metric_area_sqm(bshape.difference(pshape))
                    if outside_area_m2 <= MIN_CROSSING_AREA_M2:
                        continue
                    review = _review_fields(
                        warning_type, "outside_area_m2", outside_area_m2,
                        evidence={"outside_area_m2": outside_area_m2},
                    )
                    explanation = (
                        SYNTHETIC_PARCEL_CROSSING_EXPLANATION.format(bld_id=bid, parcel_id=pid)
                        if is_synthetic else
                        REAL_PARCEL_CROSSING_EXPLANATION.format(bld_id=bid, parcel_id=pid)
                    )
                    warnings.append(TopologyWarning(
                        warning_id=f"W-PARCEL-CROSS-{bid}-{pid}",
                        warning_type=warning_type,
                        feature_ids=[bid, pid],
                        explanation=explanation,
                        plain_language_rule="Building footprint should be wholly inside parcel boundary without crossing parcel edge.",
                        source=bsrc,
                        status="open",
                        affected_coordinates=[inter.centroid.x, inter.centroid.y] if not inter.is_empty else None,
                        **review,
                    ))
            except Exception:
                continue

    return warnings


def check_buildings_outside_parcels(
    buildings: List[Dict[str, Any]], parcels: List[Dict[str, Any]]
) -> List[TopologyWarning]:
    """Flag buildings beyond the configured distance from any loaded parcel."""
    warnings: List[TopologyWarning] = []
    if not parcels:
        return warnings
    parsed_parcels = [
        (p.get("id") or p.get("properties", {}).get("feature_id", "UNKNOWN"), safe_parse_geometry(p.get("geometry")))
        for p in parcels
    ]
    parsed_parcels = [(pid, shape) for pid, shape in parsed_parcels if shape is not None and shape.is_valid]
    if not parsed_parcels:
        return warnings

    for building in buildings:
        bid = building.get("id") or building.get("properties", {}).get("feature_id", "UNKNOWN")
        shape = safe_parse_geometry(building.get("geometry"))
        if shape is None or not shape.is_valid:
            continue
        if any(shape.intersects(parcel) for _, parcel in parsed_parcels):
            continue
        nearest_id, distance_m = min(
            ((pid, calculate_metric_distance_meters(shape, parcel)) for pid, parcel in parsed_parcels),
            key=lambda item: item[1],
        )
        if distance_m <= MIN_CROSSING_DEPTH_M:
            continue
        review = _review_fields(
            "building_outside_parcel", "distance_to_parcel_m", distance_m,
            evidence={"nearest_parcel_id": nearest_id, "distance_to_parcel_m": distance_m},
        )
        warnings.append(TopologyWarning(
            warning_id=f"W-OUTSIDE-PARCEL-{bid}",
            warning_type="building_outside_parcel",
            feature_ids=[bid, nearest_id],
            explanation=f"Building footprint {bid} is {distance_m:.2f} m from the nearest loaded parcel boundary {nearest_id}.",
            plain_language_rule="A building should intersect a parcel when the loaded parcel layer covers this area.",
            source=building.get("properties", {}).get("source", "project_vaayu_sample"),
            affected_coordinates=[shape.centroid.x, shape.centroid.y],
            **review,
        ))
    return warnings


def check_low_confidence_detections(features: List[Dict[str, Any]]) -> List[TopologyWarning]:
    warnings: List[TopologyWarning] = []
    for feature in features:
        properties = feature.get("properties", {})
        source = str(properties.get("source", ""))
        confidence = properties.get("confidence")
        if "ai" not in source.lower() or confidence is None:
            continue
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue
        if confidence >= LOW_CONFIDENCE_THRESHOLD:
            continue
        feature_id = feature.get("id") or properties.get("feature_id", "UNKNOWN")
        review = _review_fields(
            "low_confidence_detection", "confidence", confidence,
            evidence={"confidence_threshold": LOW_CONFIDENCE_THRESHOLD},
            lower_is_worse=True,
            detection_confidence=confidence,
        )
        warnings.append(TopologyWarning(
            warning_id=f"W-LOW-CONFIDENCE-{feature_id}",
            warning_type="low_confidence_detection",
            feature_ids=[feature_id],
            explanation=f"AI detection confidence is {confidence:.2f}, below the {LOW_CONFIDENCE_THRESHOLD:.2f} review threshold.",
            plain_language_rule="AI detections below the configured confidence threshold require visual confirmation.",
            source=source,
            **review,
        ))
    return warnings

def check_synthetic_parcel_crossings(synthetic_features: List[Dict[str, Any]]) -> List[TopologyWarning]:
    """Checks if synthetic building crosses synthetic parcel polygon boundary."""
    synth_buildings = [
        f for f in synthetic_features 
        if f.get("properties", {}).get("feature_type") == "synthetic_building"
    ]
    synth_parcels = [
        f for f in synthetic_features 
        if f.get("properties", {}).get("feature_type") == "synthetic_parcel"
    ]
    return check_building_parcel_crossings(synth_buildings, synth_parcels, is_synthetic=True)

def run_full_topology_validation(
    buildings: List[Dict[str, Any]],
    roads: List[Dict[str, Any]],
    real_parcels: List[Dict[str, Any]],
    synthetic_features: List[Dict[str, Any]]
) -> List[TopologyWarning]:
    """
    Executes all topology checks across loaded layers.
    GUARANTEE: If real_parcels has 0 features, zero real parcel conflict warnings are produced.
    """
    all_warnings: List[TopologyWarning] = []
    
    # 1. Individual geometry validity checks
    all_warnings.extend(validate_feature_geometries(buildings, "project_vaayu_sample"))
    all_warnings.extend(validate_feature_geometries(roads, "project_vaayu_sample"))
    all_warnings.extend(validate_feature_geometries(synthetic_features, "synthetic_test"))
    all_warnings.extend(check_low_confidence_detections(buildings))
    
    # 2. Polygon overlaps (within building footprint layer)
    all_warnings.extend(check_polygon_overlaps(buildings, "project_vaayu_sample"))
    all_warnings.extend(check_polygon_overlaps(synthetic_features, "synthetic_test"))
    
    # 3. Building-road spatial intersections
    all_warnings.extend(check_building_road_intersections(buildings, roads))
    
    # Synthetic road corridors & buildings
    synth_buildings = [f for f in synthetic_features if f.get("properties", {}).get("feature_type") == "synthetic_building"]
    synth_roads = [f for f in synthetic_features if f.get("properties", {}).get("feature_type") == "synthetic_road_corridor"]
    if synth_buildings and synth_roads:
        all_warnings.extend(check_building_road_intersections(synth_buildings, synth_roads))
        
    # 4. Synthetic parcel crossings
    all_warnings.extend(check_synthetic_parcel_crossings(synthetic_features))
    
    # 5. Real parcel crossings: GATED ON NON-EMPTY PARCELS!
    if real_parcels and len(real_parcels) > 0:
        all_warnings.extend(check_building_parcel_crossings(buildings, real_parcels, is_synthetic=False))
        all_warnings.extend(check_buildings_outside_parcels(buildings, real_parcels))
        
    return all_warnings
