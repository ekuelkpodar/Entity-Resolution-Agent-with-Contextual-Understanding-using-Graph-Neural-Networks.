"""
Beneficial ownership resolution and analysis
"""
from typing import List, Optional, Set, Tuple
from datetime import datetime

from src.entity_resolution.core.entities import (
    Entity,
    EntityType,
    UBO,
    OwnershipChain,
    CircularPattern,
    NomineeFlag,
    Relationship
)
from src.entity_resolution.core.graph_db import Neo4jConnection
from config.settings import settings


class BeneficialOwnershipResolver:
    """
    Resolve ultimate beneficial ownership through complex corporate structures
    """

    def __init__(
        self,
        graph_db: Neo4jConnection,
        threshold_pct: Optional[float] = None
    ):
        """
        Initialize UBO resolver

        Args:
            graph_db: Neo4j database connection
            threshold_pct: Ownership threshold for UBO (default from settings)
        """
        self.graph_db = graph_db
        self.threshold_pct = threshold_pct or settings.ubo_ownership_threshold

    async def find_ultimate_beneficial_owners(
        self,
        entity_id: str,
        max_depth: Optional[int] = None
    ) -> List[UBO]:
        """
        Traverse ownership graph to find UBOs

        Args:
            entity_id: Starting entity
            max_depth: Maximum ownership chain depth to traverse

        Returns:
            List of ultimate beneficial owners
        """
        max_depth = max_depth or settings.ubo_max_depth

        # Cypher query to find ownership paths
        query = f"""
        MATCH path = (start:Entity {{entity_id: $entity_id}})<-[r:RELATIONSHIP*1..{max_depth}]-(owner:Entity)
        WHERE owner.entity_type IN ['INDIVIDUAL', 'GOVERNMENT']
        AND all(rel IN r WHERE rel.relationship_type IN ['OWNS', 'CONTROLS', 'UBO_OF', 'SHAREHOLDER_OF'])
        WITH path, owner,
             reduce(pct = 100.0, rel IN relationships(path) |
                pct * COALESCE(rel.properties.ownership_pct, 100.0) / 100.0
             ) AS effective_ownership_pct
        WHERE effective_ownership_pct >= $threshold
        RETURN DISTINCT owner, effective_ownership_pct, path
        ORDER BY effective_ownership_pct DESC
        LIMIT 50
        """

        results = self.graph_db.execute_query(
            query,
            {
                "entity_id": entity_id,
                "threshold": self.threshold_pct
            }
        )

        ubos = []
        for record in results:
            owner_data = record["owner"]
            ownership_pct = record["effective_ownership_pct"]

            # Convert Neo4j node to Entity
            owner = self._neo4j_to_entity(owner_data)

            # Create ownership chain
            # Note: path extraction from Neo4j would go here
            # For now, create a basic chain
            ownership_chain = OwnershipChain(
                effective_ownership_pct=ownership_pct,
                chain_length=1  # Simplified
            )

            ubo = UBO(
                entity=owner,
                effective_ownership_pct=ownership_pct,
                ownership_chain=ownership_chain,
                confidence=self._calculate_ubo_confidence(ownership_pct, 1)
            )

            ubos.append(ubo)

        return ubos

    async def detect_circular_ownership(self, entity_id: str) -> List[CircularPattern]:
        """
        Detect circular ownership patterns (A owns B owns C owns A)

        Args:
            entity_id: Entity to check for circular patterns

        Returns:
            List of circular patterns found
        """
        query = """
        MATCH path = (start:Entity {entity_id: $entity_id})-[r:RELATIONSHIP*]->(start)
        WHERE all(rel IN r WHERE rel.relationship_type IN ['OWNS', 'CONTROLS', 'SHAREHOLDER_OF'])
        AND length(path) > 1
        RETURN path
        LIMIT 10
        """

        results = self.graph_db.execute_query(query, {"entity_id": entity_id})

        circular_patterns = []
        for record in results:
            # Extract entity IDs from path
            # This would need actual path parsing from Neo4j
            pattern = CircularPattern(
                entities_involved=[entity_id],
                risk_level="HIGH",
                description="Circular ownership detected"
            )
            circular_patterns.append(pattern)

        return circular_patterns

    async def detect_nominee_arrangements(self, entity_id: str) -> List[NomineeFlag]:
        """
        Detect potential nominee director/shareholder arrangements

        Args:
            entity_id: Entity to check

        Returns:
            List of nominee flags
        """
        threshold = settings.nominee_relationship_threshold

        query = """
        MATCH (entity:Entity {entity_id: $entity_id})<-[r:RELATIONSHIP]-(person:Entity)
        WHERE person.entity_type = 'INDIVIDUAL'
        AND r.relationship_type IN ['DIRECTOR_OF', 'SHAREHOLDER_OF']
        WITH person
        MATCH (person)-[r2:RELATIONSHIP]->(other:Entity)
        WHERE r2.relationship_type IN ['DIRECTOR_OF', 'SHAREHOLDER_OF']
        WITH person, count(DISTINCT other) as relationship_count, collect(DISTINCT other.name) as entities
        WHERE relationship_count >= $threshold
        RETURN person, relationship_count, entities
        """

        results = self.graph_db.execute_query(
            query,
            {
                "entity_id": entity_id,
                "threshold": threshold
            }
        )

        nominee_flags = []
        for record in results:
            person_data = record["person"]
            count = record["relationship_count"]
            entities = record["entities"]

            person = self._neo4j_to_entity(person_data)

            flag = NomineeFlag(
                individual=person,
                entity_count=count,
                entities_involved=entities,
                risk_level="HIGH" if count > 10 else "MEDIUM",
                rationale=f"Individual is director/shareholder of {count} entities"
            )

            nominee_flags.append(flag)

        return nominee_flags

    def _calculate_ubo_confidence(
        self,
        ownership_pct: float,
        chain_length: int
    ) -> float:
        """
        Calculate confidence in UBO identification

        Args:
            ownership_pct: Effective ownership percentage
            chain_length: Length of ownership chain

        Returns:
            Confidence score (0-1)
        """
        confidence = 1.0

        # Reduce confidence for long ownership chains
        if chain_length > 3:
            confidence *= 0.9 ** (chain_length - 3)

        # Reduce confidence for low ownership percentages
        if ownership_pct < 50:
            confidence *= (ownership_pct / 50)

        # Reduce confidence for very low ownership
        if ownership_pct < 25:
            confidence *= 0.8

        return max(0.0, min(1.0, confidence))

    def _neo4j_to_entity(self, node_data: dict) -> Entity:
        """
        Convert Neo4j node to Entity object

        Args:
            node_data: Neo4j node properties

        Returns:
            Entity object
        """
        return Entity(
            entity_id=node_data.get("entity_id", ""),
            name=node_data.get("name", ""),
            original_name=node_data.get("original_name", ""),
            entity_type=EntityType(node_data.get("entity_type", "COMPANY")),
            identifiers=node_data.get("identifiers", {}),
            addresses=node_data.get("addresses", []),
            jurisdiction=node_data.get("jurisdiction"),
            confidence=node_data.get("confidence", 1.0),
        )

    async def analyze_ownership_structure(
        self,
        entity_id: str
    ) -> dict:
        """
        Comprehensive analysis of ownership structure

        Args:
            entity_id: Entity to analyze

        Returns:
            Analysis dictionary
        """
        # Find UBOs
        ubos = await self.find_ultimate_beneficial_owners(entity_id)

        # Detect circular patterns
        circular_patterns = await self.detect_circular_ownership(entity_id)

        # Detect nominee arrangements
        nominee_flags = await self.detect_nominee_arrangements(entity_id)

        # Calculate complexity metrics
        complexity_score = self._calculate_complexity_score(
            len(ubos),
            len(circular_patterns),
            len(nominee_flags)
        )

        return {
            "entity_id": entity_id,
            "ubos": [
                {
                    "name": ubo.entity.name,
                    "ownership_pct": ubo.effective_ownership_pct,
                    "confidence": ubo.confidence
                }
                for ubo in ubos
            ],
            "circular_patterns": len(circular_patterns),
            "nominee_flags": len(nominee_flags),
            "complexity_score": complexity_score,
            "risk_level": self._determine_risk_level(complexity_score),
            "transparency": self._assess_transparency(ubos, complexity_score)
        }

    def _calculate_complexity_score(
        self,
        num_ubos: int,
        num_circular: int,
        num_nominees: int
    ) -> float:
        """
        Calculate ownership structure complexity score

        Args:
            num_ubos: Number of UBOs
            num_circular: Number of circular patterns
            num_nominees: Number of nominee flags

        Returns:
            Complexity score (0-1, higher is more complex)
        """
        # Base complexity from number of UBOs
        ubo_complexity = min(num_ubos / 5.0, 1.0)

        # Add complexity for circular patterns
        circular_complexity = min(num_circular / 3.0, 1.0) * 0.5

        # Add complexity for nominees
        nominee_complexity = min(num_nominees / 5.0, 1.0) * 0.3

        total_complexity = min(
            ubo_complexity + circular_complexity + nominee_complexity,
            1.0
        )

        return total_complexity

    def _determine_risk_level(self, complexity_score: float) -> str:
        """
        Determine risk level based on complexity

        Args:
            complexity_score: Complexity score (0-1)

        Returns:
            Risk level string
        """
        if complexity_score >= 0.7:
            return "HIGH"
        elif complexity_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _assess_transparency(self, ubos: List[UBO], complexity_score: float) -> str:
        """
        Assess ownership transparency

        Args:
            ubos: List of UBOs
            complexity_score: Complexity score

        Returns:
            Transparency assessment
        """
        if not ubos:
            return "OPAQUE"

        # High confidence UBOs with low complexity = transparent
        if complexity_score < 0.3 and all(ubo.confidence > 0.8 for ubo in ubos):
            return "TRANSPARENT"

        # Medium complexity or confidence
        if complexity_score < 0.6:
            return "PARTIALLY_TRANSPARENT"

        return "OPAQUE"


class OwnershipNetworkAnalyzer:
    """
    Analyze ownership networks for suspicious patterns
    """

    def __init__(self, graph_db: Neo4jConnection):
        """
        Initialize network analyzer

        Args:
            graph_db: Neo4j database connection
        """
        self.graph_db = graph_db

    async def find_layered_structures(
        self,
        entity_id: str,
        min_layers: int = 3
    ) -> List[List[Entity]]:
        """
        Find layered ownership structures

        Args:
            entity_id: Entity to analyze
            min_layers: Minimum number of layers to flag

        Returns:
            List of ownership chains with excessive layering
        """
        query = f"""
        MATCH path = (start:Entity {{entity_id: $entity_id}})<-[r:RELATIONSHIP*{min_layers}..10]-(owner:Entity)
        WHERE all(rel IN r WHERE rel.relationship_type IN ['OWNS', 'CONTROLS'])
        RETURN path
        LIMIT 20
        """

        results = self.graph_db.execute_query(query, {"entity_id": entity_id})

        layered_structures = []
        # Parse paths from results
        # This would require proper Neo4j path parsing

        return layered_structures

    async def find_common_controllers(
        self,
        entity_ids: List[str]
    ) -> List[Entity]:
        """
        Find entities that control multiple specified entities

        Args:
            entity_ids: List of entity IDs to check

        Returns:
            List of common controllers
        """
        query = """
        MATCH (controller:Entity)-[r:RELATIONSHIP]->(controlled:Entity)
        WHERE controlled.entity_id IN $entity_ids
        AND r.relationship_type IN ['OWNS', 'CONTROLS', 'UBO_OF']
        WITH controller, count(DISTINCT controlled) as control_count
        WHERE control_count >= 2
        RETURN controller, control_count
        ORDER BY control_count DESC
        """

        results = self.graph_db.execute_query(query, {"entity_ids": entity_ids})

        controllers = []
        for record in results:
            controller_data = record["controller"]
            # Convert to Entity
            # controllers.append(entity)

        return controllers
