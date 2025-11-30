"""
Pydantic response models for API
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Detection(BaseModel):
    """Single euphemism detection"""
    
    type: str = Field(..., description="Type of euphemism pattern")
    text: str = Field(..., description="Detected euphemism text")
    start: int = Field(..., description="Start position in original text")
    end: int = Field(..., description="End position in original text")
    confidence: float = Field(..., description="Detection confidence score")
    method: str = Field(..., description="Detection method used")
    entity: str = Field(..., description="Resolved entity")
    entity_confidence: float = Field(..., description="Entity resolution confidence")
    alternatives: List[Dict[str, Any]] = Field(
        default=[],
        description="Alternative entity resolutions"
    )


class AnalysisResponse(BaseModel):
    """Response model for text analysis"""
    
    text: str = Field(..., description="Original input text")
    detections: List[Detection] = Field(..., description="List of detected euphemisms")
    total_detected: int = Field(..., description="Total number of detections")
    processing_time: float = Field(..., description="Processing time in seconds")
    stages: Dict[str, int] = Field(..., description="Detection counts by stage")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "text": "S    ",
                "detections": [
                    {
                        "type": "company_anonymized",
                        "text": "S ",
                        "start": 0,
                        "end": 6,
                        "confidence": 0.95,
                        "method": "pattern_matching",
                        "entity": "",
                        "entity_confidence": 0.92,
                        "alternatives": [
                            {"entity": "SK", "confidence": 0.78}
                        ]
                    },
                    {
                        "type": "country_reference",
                        "text": " ",
                        "start": 9,
                        "end": 13,
                        "confidence": 0.95,
                        "method": "pattern_matching",
                        "entity": "",
                        "entity_confidence": 0.88,
                        "alternatives": []
                    }
                ],
                "total_detected": 2,
                "processing_time": 0.15,
                "stages": {
                    "pattern_matching": 2,
                    "ml_detection": 0,
                    "resolved": 2
                },
                "timestamp": "2025-10-13T12:00:00"
            }
        }


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis"""
    
    results: List[AnalysisResponse] = Field(..., description="Analysis results for each text")
    total_texts: int = Field(..., description="Total number of texts analyzed")
    total_detections: int = Field(..., description="Total detections across all texts")
    processing_time: float = Field(..., description="Total processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether ML models are loaded")
    trained_model: Optional[bool] = Field(default=False, description="Whether using trained (fine-tuned) model")
    gemini_enabled: Optional[bool] = Field(default=False, description="Whether Gemini AI explanation is enabled")
    database: Optional[str] = Field(default=None, description="Database connection status")
    cache: Optional[str] = Field(default=None, description="Cache connection status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StatisticsResponse(BaseModel):
    """Statistics response"""
    
    total_requests: int = Field(..., description="Total API requests")
    total_detections: int = Field(..., description="Total euphemisms detected")
    average_processing_time: float = Field(..., description="Average processing time")
    models_info: Dict[str, Any] = Field(..., description="Model information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response"""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid request",
                "detail": "Text cannot be empty",
                "timestamp": "2025-10-13T12:00:00"
            }
        }


class AnnotationResponse(BaseModel):
    """Response for annotation submission"""

    annotation_id: str = Field(..., description="Unique annotation identifier")
    text: str = Field(..., description="Annotated text")
    euphemism_text: str = Field(..., description="The annotated euphemism")
    resolved_entity: str = Field(..., description="Resolved entity")
    status: str = Field(default="saved", description="Annotation status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "annotation_id": "ann_123456789",
                "text": "S    ",
                "euphemism_text": "S ",
                "resolved_entity": "",
                "status": "saved",
                "timestamp": "2025-11-20T12:00:00"
            }
        }


class BatchAnnotationResponse(BaseModel):
    """Response for batch annotation upload"""

    saved_count: int = Field(..., description="Number of annotations saved")
    failed_count: int = Field(default=0, description="Number of failed annotations")
    annotation_ids: List[str] = Field(..., description="List of created annotation IDs")
    errors: List[Dict[str, str]] = Field(default=[], description="List of errors if any")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnnotationStatsResponse(BaseModel):
    """Statistics about annotation dataset"""

    total_annotations: int = Field(..., description="Total annotation count")
    by_type: Dict[str, int] = Field(..., description="Count by euphemism type")
    by_entity: Dict[str, int] = Field(..., description="Count by resolved entity")
    avg_confidence: float = Field(..., description="Average annotator confidence")
    annotators: List[str] = Field(..., description="List of annotator IDs")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
