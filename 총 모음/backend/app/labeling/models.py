"""
Pydantic models for labeling API
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class EuphemismType(str, Enum):
    """   """
    COMPANY_ANONYMIZED = "company_anonymized"  # S 
    PERSON_INITIAL = "person_initial"  # K
    COUNTRY_REFERENCE = "country_reference"  #  
    INITIAL_COMPANY = "initial_company"  # S
    OTHER = "other"


class LabelingTask(BaseModel):
    """  """
    task_id: str
    text: str
    suggested_euphemism: Optional[str] = None
    suggested_entity: Optional[str] = None
    confidence: Optional[float] = None
    context: Optional[str] = None
    category: str
    source: str  # 'aihub', 'namuwiki'
    created_at: datetime = Field(default_factory=datetime.now)


class LabelData(BaseModel):
    """   """
    task_id: str
    labeler_id: str
    euphemism_text: str
    start_pos: int = Field(ge=0)
    end_pos: int = Field(ge=0)
    euphemism_type: EuphemismType
    resolved_entity: str
    confidence: float = Field(ge=0.0, le=1.0)
    notes: Optional[str] = None
    labeling_time_seconds: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class LabelingResponse(BaseModel):
    """  """
    success: bool
    task_id: str
    message: str


class LabelingStatistics(BaseModel):
    """ """
    total_labeled: int
    total_pending: int
    by_labeler: dict
    by_category: dict
    quality_score: Optional[float] = None  # Cohen's Kappa
    estimated_completion_days: Optional[int] = None


class TaskFilter(BaseModel):
    """  """
    category: Optional[str] = None
    source: Optional[str] = None
    has_suggestion: Optional[bool] = None
