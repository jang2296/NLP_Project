"""
User 
API     
"""

import secrets
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship

from backend.app.core.database import Base


class User(Base):
    """  DB """

    __tablename__ = "users"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # User Information
    email = Column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
        comment="  "
    )

    # API Key Authentication
    api_key = Column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
        comment="API   (ked_  )"
    )

    # Account Status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="  "
    )

    # Rate Limiting
    request_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="   API  "
    )

    request_limit = Column(
        Integer,
        default=1000,
        nullable=False,
        comment="    "
    )

    # Timestamps
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        comment="   (UTC)"
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        comment="   (UTC)"
    )

    # Relationships
    analysis_logs = relationship(
        "AnalysisLog",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, active={self.is_active})>"

    @staticmethod
    def generate_api_key() -> str:
        """API   (ked_ )"""
        token = secrets.token_urlsafe(32)
        return f"ked_{token}"

    def is_rate_limited(self) -> bool:
        """   """
        return self.request_count >= self.request_limit

    def increment_request_count(self) -> None:
        """  1 """
        self.request_count += 1

    def reset_request_count(self) -> None:
        """   ( )"""
        self.request_count = 0

    def deactivate(self) -> None:
        """ """
        self.is_active = False

    def activate(self) -> None:
        """ """
        self.is_active = True

    def update_rate_limit(self, new_limit: int) -> None:
        """  """
        if new_limit <= 0:
            raise ValueError("  ")
        self.request_limit = new_limit

    def to_dict(self) -> dict:
        """  (API  )"""
        return {
            "id": self.id,
            "email": self.email,
            "is_active": self.is_active,
            "request_count": self.request_count,
            "request_limit": self.request_limit,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
