"""Configurable review thresholds for topology warnings.

These are project review thresholds chosen by the review team to help prioritize
triage. They are not legal or cadastral standards and should be tuned in the same
way as other review and QA controls.
"""

MIN_OVERLAP_AREA_M2 = 1.0
MIN_CROSSING_DEPTH_M = 0.5
MIN_CROSSING_AREA_M2 = 0.5
LOW_CONFIDENCE_THRESHOLD = 0.70

# Severity cut-offs are based on how far the measured value exceeds the threshold.
# The ratio is: measured_value / tolerance_value, with a value of 1.0 meaning equal to the tolerance.
# These are review thresholds, not legal standards.
SEVERITY_RATIO_CUTOFFS = {
    "low": 1.0,
    "medium": 2.0,
    "high": 5.0,
}

TOLERANCE_LOOKUP = {
    "building_crosses_parcel": {
        "name": "outside_area_m2",
        "value": MIN_CROSSING_AREA_M2,
        "unit": "m2",
    },
    "building_crosses_synthetic_parcel": {
        "name": "outside_area_m2",
        "value": MIN_CROSSING_AREA_M2,
        "unit": "m2",
    },
    "parcel_overlap": {
        "name": "overlap_area_m2",
        "value": MIN_OVERLAP_AREA_M2,
        "unit": "m2",
    },
    "overlapping_polygons": {
        "name": "overlap_area_m2",
        "value": MIN_OVERLAP_AREA_M2,
        "unit": "m2",
    },
    "building_road_spatial_overlap": {
        "name": "overlap_area_m2",
        "value": 0.0,
        "unit": "m2",
    },
    "low_confidence_detection": {
        "name": "confidence_threshold",
        "value": LOW_CONFIDENCE_THRESHOLD,
        "unit": "ratio",
    },
    "building_outside_parcel": {
        "name": "distance_to_parcel_m",
        "value": MIN_CROSSING_DEPTH_M,
        "unit": "m",
    },
    "empty_geometry": {
        "name": "has_coordinates",
        "value": 1.0,
        "unit": "boolean",
    },
    "invalid_geometry": {
        "name": "geometry_validity",
        "value": 0.0,
        "unit": "ratio",
    },
}

SUGGESTED_ACTION_LOOKUP = {
    "building_crosses_parcel": {
        "action_type": "field_visit",
        "text": "Verify on site whether a subdivision or boundary change was recorded",
    },
    "parcel_overlap": {
        "action_type": "desk_review",
        "text": "Reconcile the two parcel records; likely a survey or digitisation error",
    },
    "overlapping_polygons": {
        "action_type": "desk_review",
        "text": "Review the intersecting footprints against the imagery and reconcile their boundaries",
    },
    "building_road_spatial_overlap": {
        "action_type": "visual_review",
        "text": "Review the building and road-corridor overlap against imagery; the corridor is not a legal right-of-way",
    },
    "building_crosses_synthetic_parcel": {
        "action_type": "desk_review",
        "text": "Inspect the synthetic demonstration boundary; this fixture is not a real parcel record",
    },
    "invalid_geometry": {
        "action_type": "desk_review",
        "text": "Repair the geometry in GIS; no field visit needed",
    },
    "empty_geometry": {
        "action_type": "desk_review",
        "text": "Restore or remove the feature geometry in GIS; no field visit needed",
    },
    "building_outside_parcel": {
        "action_type": "mapping_initiation",
        "text": "No parcel record found; initiate mapping for this area (not an enforcement action)",
    },
    "low_confidence_detection": {
        "action_type": "desk_review",
        "text": "Check the detection against the imagery; confirm or reject",
    },
}
