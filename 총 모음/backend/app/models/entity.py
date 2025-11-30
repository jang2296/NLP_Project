"""
Entity Model - Knowledge Base  

    :
-    (, ,  )
-     
- (aliases) 
-   (  )
"""

from datetime import datetime
from typing import List, Optional, Dict

from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, Index, Float, Boolean
from sqlalchemy.dialects.postgresql import ARRAY

from backend.app.core.database import Base


class Entity(Base):
    """
    (Entity)  - Knowledge Base 

    Attributes:
        id:   ID (Primary Key)
        name:   (Unique, Indexed)
        type:   (company, person, country, organization)
        description:   ( )
        aliases:   (JSON )
        embedding:   (768, Sentence Transformer)
        frequency:   ()
        confidence_threshold:     
        is_active:  
        metadata:   (JSON)
        created_at:  
        updated_at:  
    """

    __tablename__ = "entities"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Entity Information
    name = Column(
        String(200),
        unique=True,
        index=True,
        nullable=False,
        comment="  (: , )"
    )

    type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="  (company, person, country, organization)"
    )

    description = Column(
        Text,
        nullable=False,
        comment="    "
    )

    # Aliases and Variations
    aliases = Column(
        JSON,
        default=list,
        nullable=False,
        comment="  (JSON )"
    )

    # Embedding for Semantic Search
    embedding = Column(
        JSON,
        nullable=True,
        comment="  (768, float )"
    )

    # Statistics
    frequency = Column(
        Integer,
        default=0,
        nullable=False,
        comment="  ()"
    )

    confidence_threshold = Column(
        Float,
        default=0.85,
        nullable=False,
        comment="  "
    )

    # Status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment=" "
    )

    # Additional Metadata
    metadata = Column(
        JSON,
        nullable=True,
        comment="  (JSON)"
    )

    # Timestamps
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        comment="  (UTC)"
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        comment="  (UTC)"
    )

    # Indexes
    __table_args__ = (
        Index('idx_entity_type_active', 'type', 'is_active'),
        Index('idx_entity_frequency', 'frequency'),
    )

    def __repr__(self) -> str:
        """ """
        return f"<Entity(id={self.id}, name={self.name}, type={self.type})>"

    @staticmethod
    def create_entity(
        name: str,
        type: str,
        description: str,
        aliases: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        confidence_threshold: float = 0.85,
        metadata: Optional[Dict] = None
    ) -> "Entity":
        """
          

        Args:
            name:  
            type:   (company, person, country, organization)
            description: 
            aliases:  
            embedding:  
            confidence_threshold:  
            metadata:  

        Returns:
            Entity:   

        Raises:
            ValueError:   type  
        """
        valid_types = {"company", "person", "country", "organization", "other"}
        if type not in valid_types:
            raise ValueError(f"Invalid entity type. Must be one of {valid_types}")

        return Entity(
            name=name,
            type=type,
            description=description,
            aliases=aliases or [],
            embedding=embedding,
            confidence_threshold=confidence_threshold,
            metadata=metadata or {}
        )

    def add_alias(self, alias: str) -> None:
        """
         

        Args:
            alias:  
        """
        if alias not in self.aliases:
            self.aliases.append(alias)

    def remove_alias(self, alias: str) -> bool:
        """
         

        Args:
            alias:  

        Returns:
            bool:   
        """
        if alias in self.aliases:
            self.aliases.remove(alias)
            return True
        return False

    def increment_frequency(self) -> None:
        """  """
        self.frequency += 1

    def update_embedding(self, embedding: List[float]) -> None:
        """
          

        Args:
            embedding:    (768)

        Raises:
            ValueError:  768  
        """
        if len(embedding) != 768:
            raise ValueError("Embedding must be 768-dimensional")
        self.embedding = embedding

    def update_description(self, description: str) -> None:
        """
         

        Args:
            description:  
        """
        self.description = description

    def deactivate(self) -> None:
        """ """
        self.is_active = False

    def activate(self) -> None:
        """ """
        self.is_active = True

    def to_dict(self) -> dict:
        """
          (API )

        embedding   .

        Returns:
            dict:   
        """
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "aliases": self.aliases,
            "frequency": self.frequency,
            "confidence_threshold": self.confidence_threshold,
            "is_active": self.is_active,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def to_dict_with_embedding(self) -> dict:
        """
           

        Returns:
            dict:    
        """
        data = self.to_dict()
        data["embedding"] = self.embedding
        return data

    def matches_alias(self, text: str) -> bool:
        """
           

        Args:
            text:  

        Returns:
            bool:   True
        """
        normalized_text = text.lower().strip()
        normalized_name = self.name.lower().strip()

        if normalized_text == normalized_name:
            return True

        return any(
            normalized_text == alias.lower().strip()
            for alias in self.aliases
        )
