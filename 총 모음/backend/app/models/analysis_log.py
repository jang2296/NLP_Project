"""
AnalysisLog Model -     

    :
-     
-     
-      
-    
"""

from datetime import datetime
from typing import Optional, Dict, List

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship

from backend.app.core.database import Base


class AnalysisLog(Base):
    """
       - API    

    Attributes:
        id:   ID (Primary Key)
        user_id:   ID (Foreign Key)
        text:    ( 5000)
        text_length:   ()
        detections_count:    
        detections_data:     (JSON)
        confidence_avg:    (0.0 ~ 1.0)
        confidence_min:   
        confidence_max:   
        processing_time:   ()
        cache_hit:   
        model_version:   
        ip_address:  IP  ()
        user_agent: User-Agent  ()
        created_at:   

    Relationships:
        user:    (many-to-one)
    """

    __tablename__ = "analysis_logs"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Foreign Key
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="  ID"
    )

    # Request Data
    text = Column(
        Text,
        nullable=False,
        comment="   ( 5000)"
    )

    text_length = Column(
        Integer,
        nullable=False,
        comment="  ()"
    )

    # Detection Results
    detections_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="   "
    )

    detections_data = Column(
        JSON,
        nullable=True,
        comment="    (JSON )"
    )

    # Confidence Metrics
    confidence_avg = Column(
        Float,
        nullable=True,
        comment="   (0.0 ~ 1.0)"
    )

    confidence_min = Column(
        Float,
        nullable=True,
        comment="  "
    )

    confidence_max = Column(
        Float,
        nullable=True,
        comment="  "
    )

    # Performance Metrics
    processing_time = Column(
        Float,
        nullable=False,
        comment="  ()"
    )

    cache_hit = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="  "
    )

    # Model Information
    model_version = Column(
        String(50),
        nullable=True,
        comment="  "
    )

    # Request Metadata
    ip_address = Column(
        String(45),
        nullable=True,
        comment=" IP  (IPv4/IPv6)"
    )

    user_agent = Column(
        String(500),
        nullable=True,
        comment="User-Agent "
    )

    # Timestamp
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
        comment="   (UTC)"
    )

    # Relationships
    user = relationship(
        "User",
        back_populates="analysis_logs"
    )

    def __repr__(self) -> str:
        """ """
        return (
            f"<AnalysisLog(id={self.id}, user_id={self.user_id}, "
            f"detections={self.detections_count}, time={self.processing_time:.3f}s)>"
        )

    @staticmethod
    def create_from_result(
        user_id: int,
        text: str,
        detections: List[Dict],
        processing_time: float,
        cache_hit: bool = False,
        model_version: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> "AnalysisLog":
        """
            

        Args:
            user_id:  ID
            text:  
            detections:   
            processing_time:   ()
            cache_hit:   
            model_version:  
            ip_address: IP 
            user_agent: User-Agent

        Returns:
            AnalysisLog:   
        """
        #  
        confidences = [d.get("confidence", 0.0) for d in detections]

        return AnalysisLog(
            user_id=user_id,
            text=text[:5000],  #  5000 
            text_length=len(text.encode('utf-8')),
            detections_count=len(detections),
            detections_data=detections,
            confidence_avg=sum(confidences) / len(confidences) if confidences else None,
            confidence_min=min(confidences) if confidences else None,
            confidence_max=max(confidences) if confidences else None,
            processing_time=processing_time,
            cache_hit=cache_hit,
            model_version=model_version,
            ip_address=ip_address,
            user_agent=user_agent
        )

    def to_dict(self) -> dict:
        """
          (API )

        Returns:
            dict:   
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "text_length": self.text_length,
            "detections_count": self.detections_count,
            "confidence_avg": self.confidence_avg,
            "processing_time": self.processing_time,
            "cache_hit": self.cache_hit,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat()
        }

    def get_performance_metrics(self) -> dict:
        """
          

        Returns:
            dict:   
        """
        return {
            "processing_time": self.processing_time,
            "cache_hit": self.cache_hit,
            "text_length": self.text_length,
            "detections_per_char": self.detections_count / self.text_length if self.text_length > 0 else 0
        }

    def get_quality_metrics(self) -> dict:
        """
          

        Returns:
            dict:   
        """
        return {
            "detections_count": self.detections_count,
            "confidence_avg": self.confidence_avg,
            "confidence_min": self.confidence_min,
            "confidence_max": self.confidence_max,
            "confidence_std": self._calculate_confidence_std() if self.detections_data else None
        }

    def _calculate_confidence_std(self) -> Optional[float]:
        """  """
        if not self.detections_data or not self.confidence_avg:
            return None

        confidences = [d.get("confidence", 0.0) for d in self.detections_data]
        if len(confidences) < 2:
            return 0.0

        variance = sum((c - self.confidence_avg) ** 2 for c in confidences) / len(confidences)
        return variance ** 0.5
