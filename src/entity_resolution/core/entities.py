"""
Core entity data models for entity resolution system
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np


class EntityType(str, Enum):
    """Entity type enumeration"""
    COMPANY = "COMPANY"
    INDIVIDUAL = "INDIVIDUAL"
    GOVERNMENT = "GOVERNMENT"
    NGO = "NGO"


class RelationshipType(str, Enum):
    """Relationship type enumeration"""
    OWNS = "OWNS"
    CONTROLS = "CONTROLS"
    DIRECTOR_OF = "DIRECTOR_OF"
    SHAREHOLDER_OF = "SHAREHOLDER_OF"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    UBO_OF = "UBO_OF"
    NOMINEE_FOR = "NOMINEE_FOR"
    SHARES_ADDRESS = "SHARES_ADDRESS"
    SHARES_DIRECTOR = "SHARES_DIRECTOR"
    TRANSACTS_WITH = "TRANSACTS_WITH"
    RELATED_TO = "RELATED_TO"
    INDIRECT_OWNS = "INDIRECT_OWNS"


@dataclass
class Entity:
    """
    Normalized entity representation
    """
    # Core identifiers
    entity_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    original_name: str = ""
    entity_type: EntityType = EntityType.COMPANY

    # Identifiers
    identifiers: Dict[str, str] = field(default_factory=dict)

    # Attributes
    addresses: List[str] = field(default_factory=list)
    incorporation_date: Optional[datetime] = None
    dissolution_date: Optional[datetime] = None
    jurisdiction: Optional[str] = None
    industry_codes: List[str] = field(default_factory=list)

    # Contact information
    phone_numbers: List[str] = field(default_factory=list)
    email_addresses: List[str] = field(default_factory=list)
    websites: List[str] = field(default_factory=list)

    # For individuals
    date_of_birth: Optional[datetime] = None
    nationality: Optional[str] = None

    # Roles and relationships
    roles: List[str] = field(default_factory=list)

    # Metadata
    data_sources: List[str] = field(default_factory=list)
    confidence: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)

    # Embeddings
    name_embedding: Optional[np.ndarray] = None
    contextual_embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "original_name": self.original_name,
            "entity_type": self.entity_type.value,
            "identifiers": self.identifiers,
            "addresses": self.addresses,
            "incorporation_date": self.incorporation_date.isoformat() if self.incorporation_date else None,
            "dissolution_date": self.dissolution_date.isoformat() if self.dissolution_date else None,
            "jurisdiction": self.jurisdiction,
            "industry_codes": self.industry_codes,
            "phone_numbers": self.phone_numbers,
            "email_addresses": self.email_addresses,
            "websites": self.websites,
            "date_of_birth": self.date_of_birth.isoformat() if self.date_of_birth else None,
            "nationality": self.nationality,
            "roles": self.roles,
            "data_sources": self.data_sources,
            "confidence": self.confidence,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create entity from dictionary"""
        # Convert datetime strings back to datetime objects
        if data.get("incorporation_date"):
            data["incorporation_date"] = datetime.fromisoformat(data["incorporation_date"])
        if data.get("dissolution_date"):
            data["dissolution_date"] = datetime.fromisoformat(data["dissolution_date"])
        if data.get("date_of_birth"):
            data["date_of_birth"] = datetime.fromisoformat(data["date_of_birth"])
        if data.get("last_updated"):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])

        # Convert entity_type string to enum
        if "entity_type" in data:
            data["entity_type"] = EntityType(data["entity_type"])

        return cls(**data)


@dataclass
class Relationship:
    """
    Relationship between entities
    """
    relationship_id: str = field(default_factory=lambda: str(uuid4()))
    source_entity_id: str = ""
    target_entity_id: str = ""
    relationship_type: RelationshipType = RelationshipType.RELATED_TO

    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    data_sources: List[str] = field(default_factory=list)

    # Temporal attributes
    temporal_valid_from: Optional[datetime] = None
    temporal_valid_to: Optional[datetime] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary"""
        return {
            "relationship_id": self.relationship_id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relationship_type": self.relationship_type.value,
            "properties": self.properties,
            "confidence": self.confidence,
            "data_sources": self.data_sources,
            "temporal_valid_from": self.temporal_valid_from.isoformat() if self.temporal_valid_from else None,
            "temporal_valid_to": self.temporal_valid_to.isoformat() if self.temporal_valid_to else None,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Create relationship from dictionary"""
        # Convert datetime strings
        if data.get("temporal_valid_from"):
            data["temporal_valid_from"] = datetime.fromisoformat(data["temporal_valid_from"])
        if data.get("temporal_valid_to"):
            data["temporal_valid_to"] = datetime.fromisoformat(data["temporal_valid_to"])
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("last_updated"):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])

        # Convert relationship_type string to enum
        if "relationship_type" in data:
            data["relationship_type"] = RelationshipType(data["relationship_type"])

        return cls(**data)


@dataclass
class Document:
    """Document containing entity information"""
    document_id: str = field(default_factory=lambda: str(uuid4()))
    text: str = ""
    source: str = ""
    document_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class OwnershipChain:
    """Ownership chain from entity to UBO"""
    chain_id: str = field(default_factory=lambda: str(uuid4()))
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    effective_ownership_pct: float = 0.0
    chain_length: int = 0


@dataclass
class UBO:
    """Ultimate Beneficial Owner"""
    entity: Entity
    effective_ownership_pct: float
    ownership_chain: OwnershipChain
    confidence: float
    identified_at: datetime = field(default_factory=datetime.now)


@dataclass
class CircularPattern:
    """Circular ownership pattern"""
    pattern_id: str = field(default_factory=lambda: str(uuid4()))
    entities_involved: List[str]
    risk_level: str = "MEDIUM"
    description: str = ""


@dataclass
class NomineeFlag:
    """Potential nominee arrangement flag"""
    flag_id: str = field(default_factory=lambda: str(uuid4()))
    individual: Entity
    entity_count: int
    entities_involved: List[str]
    risk_level: str = "MEDIUM"
    rationale: str = ""


@dataclass
class StringSimilarityScores:
    """String similarity scores between entities"""
    name_similarity: float = 0.0
    address_similarity: float = 0.0
    identifier_match_score: float = 0.0


@dataclass
class GraphSignals:
    """Graph-based signals for entity matching"""
    shared_directors: List[Entity] = field(default_factory=list)
    num_shared_directors: int = 0
    shared_directors_score: float = 0.0
    shared_addresses: List[str] = field(default_factory=list)
    shared_addresses_score: float = 0.0
    common_owners: List[Entity] = field(default_factory=list)
    common_owners_score: float = 0.0
    transitive_paths: List[List[Entity]] = field(default_factory=list)
    transitive_score: float = 0.0


@dataclass
class ContextualAnalysis:
    """LLM-based contextual analysis of entity"""
    profile_summary: str = ""
    risk_indicators: List[Dict[str, Any]] = field(default_factory=list)
    beneficial_ownership: Dict[str, Any] = field(default_factory=dict)
    entity_classification: str = "UNKNOWN"
    classification_confidence: float = 0.0
    contextual_notes: str = ""
    recommended_actions: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextualAnalysis":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class DisambiguationResult:
    """Result of LLM-based entity disambiguation"""
    is_match: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    supporting_factors: List[str] = field(default_factory=list)
    contradicting_factors: List[str] = field(default_factory=list)
    recommendation: str = "NEEDS_INVESTIGATION"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisambiguationResult":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class EntityResolutionDecision:
    """Final entity resolution decision"""
    decision: str  # DEFINITE_MATCH, PROBABLE_MATCH, POSSIBLE_MATCH, NOT_MATCH
    confidence: float
    action: str  # MERGE_ENTITIES, FLAG_FOR_REVIEW, NEEDS_INVESTIGATION, KEEP_SEPARATE
    explanation: str
    contributing_signals: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "action": self.action,
            "explanation": self.explanation,
            "contributing_signals": self.contributing_signals,
            "timestamp": self.timestamp.isoformat(),
        }
