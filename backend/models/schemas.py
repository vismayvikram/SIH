"""
Standard Pydantic models for SIH26012 Shared Feature Schema and Review Workflow.
Conforms to SIH26012_Comprehensive_Project_Blueprint and schema-v1.json.
"""
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime

# Permitted source types
SourceType = Literal[
    "project_vaayu_sample",     # Reported hackathon sample, unverified
    "osm",                      # OpenStreetMap centerlines (ODbL 1.0)
    "manual_visual_reference",  # Human reviewer digitized
    "synthetic_test",           # Fabricated QA demonstration fixture
    "ai_building_model",        # AI model inference output
    "official_reference"        # Official government cadastral layer (absent in this bundle)
]

VerificationStatusType = Literal[
    "unverified",
    "survey_verified"
]

ReviewStatusType = Literal[
    "unverified",
    "under_review",
    "approved",
    "rejected"
]

FeatureType = Literal[
    "building",
    "village_road_polygon",
    "road",
    "parcel",
    "synthetic_building",
    "synthetic_road_corridor",
    "synthetic_parcel",
    "draft_polygon",
    "draft_line"
]

WarningSeverityType = Literal[
    "low",
    "medium",
    "high",
    "critical"
]

WarningType = Literal[
    "overlapping_polygons",
    "building_road_spatial_overlap",
    "building_crosses_synthetic_parcel",
    "invalid_geometry",
    "empty_geometry",
    "sliver_polygon",
    "unmapped_feature"
]

class GeoJSONGeometry(BaseModel):
    type: Literal["Point", "LineString", "Polygon", "MultiPolygon", "GeometryCollection"]
    coordinates: Any

class FeatureProperties(BaseModel):
    feature_id: str
    feature_type: str = "building"
    source: str = "project_vaayu_sample"
    source_date: Optional[str] = None
    
    # AI Model Fields
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    confidence: Optional[float] = None
    
    # Review & Verification Fields
    verification_status: VerificationStatusType = "unverified"
    review_status: ReviewStatusType = "unverified"
    review_score: Optional[int] = Field(default=None, ge=0, le=100)
    warning_ids: List[str] = Field(default_factory=list)
    parent_feature_ids: List[str] = Field(default_factory=list)
    
    # Audit & Human Edit Fields
    edited_by: Optional[str] = None
    edited_at: Optional[str] = None
    notes: str = ""
    edit_reason: Optional[str] = None
    
    # Physical Attributes (anonymized)
    area_sqm: Optional[float] = None
    roof_type: Optional[str] = None
    no_floors: Optional[int] = None
    highway: Optional[str] = None
    locality: str = "Lalpur"
    lgd_village_code: str = "511638"
    
    # Citation / Provenance
    provenance_citation: str = (
        "Project Vaayu sample; repository README describes underlying data as "
        "Ministry-supplied SVAMITVA/hackathon material. Provenance is reported, "
        "not independently verified. Not official parcel ground truth."
    )
    
    # Additional raw properties preserved without dropping
    extra: Dict[str, Any] = Field(default_factory=dict)

class ReviewFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    id: str
    geometry: GeoJSONGeometry
    properties: FeatureProperties
    original_geometry: Optional[GeoJSONGeometry] = None

class FeatureCollectionResponse(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    name: str
    crs: Dict[str, Any] = Field(default_factory=lambda: {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
    })
    total_features: int
    features: List[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ScoreRuleContribution(BaseModel):
    rule_id: str
    rule_name: str
    description: str
    points: int

class ReviewScoreBreakdown(BaseModel):
    feature_id: str
    feature_type: str
    total_score: int = Field(ge=0, le=100)
    priority: Literal["low", "medium", "high"]
    heuristic_disclaimer: str = "prototype heuristic—not a validated survey-priority model"
    rules_triggered: List[ScoreRuleContribution]

class TopologyWarning(BaseModel):
    warning_id: str
    warning_type: WarningType
    severity: WarningSeverityType
    feature_ids: List[str]
    explanation: str
    plain_language_rule: str
    source: str = "synthetic_test"
    status: Literal["open", "resolved", "ignored"] = "open"
    affected_coordinates: Optional[List[float]] = None

class GeometryEditRequest(BaseModel):
    geometry: GeoJSONGeometry
    reviewer_label: str = "demo-reviewer"
    edit_reason: str = "Boundary refinement during inspection"

class FeatureStatusUpdateRequest(BaseModel):
    review_status: ReviewStatusType
    notes: Optional[str] = None
    reviewer_label: str = "demo-reviewer"

class DraftFeatureCreateRequest(BaseModel):
    feature_type: Literal["draft_polygon", "draft_line"] = "draft_polygon"
    geometry: GeoJSONGeometry
    notes: str = "Human traced draft feature"
    reviewer_label: str = "demo-reviewer"

class ModelPredictRequest(BaseModel):
    mode: Literal["live", "mock", "precomputed", "deeplab"] = "mock"
    model_name: str = "giswqs/whu-building-unetplusplus-efficientnet-b4"
    model_version: str = "09df9efd323bbd3d56b98b4857129eb9b5baa2d3"
    confidence_threshold: float = 0.5
    simulate_failure: bool = False

class ExportMetadata(BaseModel):
    export_timestamp: str
    project_id: str = "SIH26012_INDIA_CANDIDATE_01_LALPUR"
    locality: str = "Lalpur Village, Gujarat (LGD 511638)"
    target_crs: str = "EPSG:4326 (WGS 84 / RFC 7946)"
    native_analysis_crs: str = "EPSG:3857 (Web Mercator)"
    provenance_disclaimer: str
    total_reviewed_features: int
    open_warnings_count: int
