"""
Core entity resolution components
"""
from src.entity_resolution.core.entities import (
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
    Document,
    UBO,
    EntityResolutionDecision
)
from src.entity_resolution.core.entity_extractor import EntityExtractor
from src.entity_resolution.core.entity_normalizer import EntityNormalizer
from src.entity_resolution.core.string_matcher import StringSimilarityMatcher
from src.entity_resolution.core.graph_db import Neo4jConnection
from src.entity_resolution.core.decision_engine import EntityResolutionDecisionEngine
from src.entity_resolution.core.beneficial_ownership import BeneficialOwnershipResolver
from src.entity_resolution.core.contextual_reasoning import ContextualReasoningEngine

__all__ = [
    "Entity",
    "Relationship",
    "EntityType",
    "RelationshipType",
    "Document",
    "UBO",
    "EntityResolutionDecision",
    "EntityExtractor",
    "EntityNormalizer",
    "StringSimilarityMatcher",
    "Neo4jConnection",
    "EntityResolutionDecisionEngine",
    "BeneficialOwnershipResolver",
    "ContextualReasoningEngine",
]
