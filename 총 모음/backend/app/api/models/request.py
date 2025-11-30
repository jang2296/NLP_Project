"""
Pydantic request models for API validation
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import html
import re


def sanitize_text(text: str) -> str:
    """
    Sanitize input text to prevent XSS and injection attacks

    Args:
        text: Raw input text

    Returns:
        str: Sanitized text
    """
    if not text:
        return text

    # Remove null bytes
    text = text.replace('\x00', '')

    # Escape HTML special characters
    text = html.escape(text, quote=False)

    # Remove potentially dangerous patterns while preserving Korean text
    # Allow Korean, numbers, common punctuation, and whitespace
    text = re.sub(r'[^\w\s---0-9.,!?()"\'\-~]', '', text)

    # Limit consecutive whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


class AnalysisRequest(BaseModel):
    """Request model for text analysis"""

    text: str = Field(
        ...,
        description="Text to analyze for euphemisms",
        min_length=1,
        max_length=10000
    )

    options: Optional[dict] = Field(
        default=None,
        description="Additional analysis options"
    )

    @validator('text')
    def validate_text(cls, v):
        """Validate and sanitize text"""
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')

        # Sanitize for security
        sanitized = sanitize_text(v)

        if not sanitized:
            raise ValueError('Text contains only invalid characters')

        # Check for excessive length after sanitization
        if len(sanitized) > 10000:
            raise ValueError('Text too long after sanitization (max 10000 chars)')

        return sanitized

    class Config:
        schema_extra = {
            "example": {
                "text": "S    ",
                "options": {}
            }
        }


class BatchAnalysisRequest(BaseModel):
    """Request model for batch text analysis"""

    texts: List[str] = Field(
        ...,
        description="List of texts to analyze",
        min_items=1,
        max_items=100
    )

    options: Optional[dict] = Field(
        default=None,
        description="Additional analysis options"
    )

    @validator('texts')
    def validate_texts(cls, v):
        """Validate and sanitize all texts"""
        if not v:
            raise ValueError('Texts list cannot be empty')

        valid_texts = []
        for text in v:
            if text and text.strip():
                # Sanitize each text
                sanitized = sanitize_text(text)

                if sanitized and len(sanitized) <= 10000:
                    valid_texts.append(sanitized)

        if not valid_texts:
            raise ValueError('No valid texts provided after sanitization')

        return valid_texts

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "S   ",
                    "    "
                ],
                "options": {}
            }
        }


class FeedbackRequest(BaseModel):
    """Request model for user feedback on detections"""

    text: str = Field(..., description="Original analyzed text")
    detection_id: str = Field(..., description="Detection identifier")
    is_correct: bool = Field(..., description="Whether detection was correct")
    correct_entity: Optional[str] = Field(None, description="Correct entity if detection was wrong")
    comment: Optional[str] = Field(None, description="Additional feedback")

    class Config:
        schema_extra = {
            "example": {
                "text": "S  ",
                "detection_id": "det_123456",
                "is_correct": True,
                "correct_entity": "",
                "comment": " "
            }
        }


class AnnotationRequest(BaseModel):
    """Request model for manual euphemism annotation"""

    text: str = Field(..., description="Text containing euphemism", min_length=1, max_length=5000)
    euphemism_text: str = Field(..., description="The euphemism phrase to annotate")
    start_pos: int = Field(..., ge=0, description="Start position of euphemism in text")
    end_pos: int = Field(..., ge=0, description="End position of euphemism in text")
    euphemism_type: str = Field(..., description="Type of euphemism (company_anonymized, person_initial, etc.)")
    resolved_entity: str = Field(..., description="The actual entity being referred to")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Annotator confidence (0-1)")
    context_notes: Optional[str] = Field(None, description="Additional context or reasoning")
    annotator_id: Optional[str] = Field(None, description="Annotator identifier")

    @validator('end_pos')
    def validate_positions(cls, v, values):
        """Validate end position is after start position"""
        if 'start_pos' in values and v <= values['start_pos']:
            raise ValueError('end_pos must be greater than start_pos')
        return v

    class Config:
        schema_extra = {
            "example": {
                "text": "S    ",
                "euphemism_text": "S ",
                "start_pos": 0,
                "end_pos": 6,
                "euphemism_type": "company_anonymized",
                "resolved_entity": "",
                "confidence": 0.95,
                "context_notes": "  ",
                "annotator_id": "annotator_001"
            }
        }


class BatchAnnotationRequest(BaseModel):
    """Request model for batch annotation upload"""

    annotations: List[AnnotationRequest] = Field(..., min_items=1, max_items=1000)
    dataset_name: Optional[str] = Field(None, description="Name of annotation dataset")

    class Config:
        schema_extra = {
            "example": {
                "annotations": [
                    {
                        "text": "S   ",
                        "euphemism_text": "S ",
                        "start_pos": 0,
                        "end_pos": 6,
                        "euphemism_type": "company_anonymized",
                        "resolved_entity": "",
                        "confidence": 1.0
                    }
                ],
                "dataset_name": "manual_annotations_v1"
            }
        }
