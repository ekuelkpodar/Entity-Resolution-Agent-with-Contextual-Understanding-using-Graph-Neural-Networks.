"""
Entity Resolution Agent with Contextual Understanding using Graph Neural Networks

A production-grade entity resolution system that uses GNNs, LLMs, and graph analysis
for accurate entity matching and beneficial ownership identification.
"""

__version__ = "0.1.0"
__author__ = "Castellum.AI Team"

from src.entity_resolution.core.entities import (
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
)

__all__ = [
    "Entity",
    "Relationship",
    "EntityType",
    "RelationshipType",
]
