"""
Entity Resolution Decision Engine
Combines multiple signals for final entity resolution decision
"""
from typing import Any, Dict, Optional

from src.entity_resolution.core.entities import (
    Entity,
    EntityResolutionDecision,
    StringSimilarityScores,
    GraphSignals,
    DisambiguationResult
)
from src.entity_resolution.core.string_matcher import StringSimilarityMatcher
from src.entity_resolution.core.contextual_reasoning import ContextualReasoningEngine
from config.settings import settings


class EntityResolutionDecisionEngine:
    """
    Combine multiple signals for entity resolution decision
    """

    def __init__(
        self,
        string_matcher: Optional[StringSimilarityMatcher] = None,
        context_engine: Optional[ContextualReasoningEngine] = None,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize decision engine

        Args:
            string_matcher: String similarity matcher
            context_engine: Contextual reasoning engine
            llm_client: LLM client for contextual analysis
        """
        self.string_matcher = string_matcher or StringSimilarityMatcher()
        self.context_engine = context_engine or ContextualReasoningEngine(llm_client)

        # Load fusion weights from settings
        self.weights = {
            "name_similarity": settings.weight_name_similarity,
            "address_similarity": settings.weight_address_similarity,
            "identifier_match": settings.weight_identifier_match,
            "gnn_embedding": settings.weight_gnn_embedding,
            "shared_directors": settings.weight_shared_directors,
            "shared_addresses": settings.weight_shared_addresses,
            "llm_confidence": settings.weight_llm_confidence,
        }

        # Decision thresholds from settings
        self.definite_match_threshold = settings.definite_match_threshold
        self.probable_match_threshold = settings.probable_match_threshold
        self.possible_match_threshold = settings.possible_match_threshold

    async def resolve_entities(
        self,
        entity1: Entity,
        entity2: Entity,
        gnn_score: Optional[float] = None,
        graph_signals: Optional[GraphSignals] = None
    ) -> EntityResolutionDecision:
        """
        Make entity resolution decision using multiple signals

        Args:
            entity1: First entity
            entity2: Second entity
            gnn_score: GNN embedding similarity score
            graph_signals: Graph pattern analysis signals

        Returns:
            EntityResolutionDecision
        """
        # Signal 1: String similarity
        string_scores = self.string_matcher.compute_similarity(entity1, entity2)

        # Signal 2: GNN embedding similarity (if available)
        if gnn_score is None:
            gnn_score = 0.0

        # Signal 3: Graph pattern analysis (if available)
        if graph_signals is None:
            graph_signals = GraphSignals()

        # Signal 4: Contextual LLM reasoning
        contextual_analysis = await self.context_engine.disambiguate_entities(
            entity1,
            entity2,
            match_confidence=gnn_score
        )

        # Fuse all signals
        final_decision = self.fuse_signals(
            string_scores,
            gnn_score,
            graph_signals,
            contextual_analysis
        )

        return final_decision

    def fuse_signals(
        self,
        string_scores: StringSimilarityScores,
        gnn_score: float,
        graph_signals: GraphSignals,
        llm_analysis: DisambiguationResult
    ) -> EntityResolutionDecision:
        """
        Combine multiple signals using weighted fusion

        Args:
            string_scores: String similarity scores
            gnn_score: GNN embedding similarity
            graph_signals: Graph pattern signals
            llm_analysis: LLM disambiguation result

        Returns:
            EntityResolutionDecision
        """
        # Weighted combination
        weighted_score = (
            self.weights["name_similarity"] * string_scores.name_similarity +
            self.weights["address_similarity"] * string_scores.address_similarity +
            self.weights["identifier_match"] * string_scores.identifier_match_score +
            self.weights["gnn_embedding"] * gnn_score +
            self.weights["shared_directors"] * graph_signals.shared_directors_score +
            self.weights["shared_addresses"] * graph_signals.shared_addresses_score +
            self.weights["llm_confidence"] * llm_analysis.confidence
        )

        # Ensure score is in valid range
        weighted_score = max(0.0, min(1.0, weighted_score))

        # Identifier match can override low scores
        if string_scores.identifier_match_score == 1.0:
            weighted_score = max(weighted_score, 0.90)

        # LLM can provide override
        if llm_analysis.is_match and llm_analysis.confidence > 0.9:
            weighted_score = max(weighted_score, 0.85)
        elif not llm_analysis.is_match and llm_analysis.confidence > 0.9:
            weighted_score = min(weighted_score, 0.40)

        # Determine decision based on thresholds
        if weighted_score >= self.definite_match_threshold:
            decision = "DEFINITE_MATCH"
            action = "MERGE_ENTITIES"
        elif weighted_score >= self.probable_match_threshold:
            decision = "PROBABLE_MATCH"
            action = "FLAG_FOR_REVIEW"
        elif weighted_score >= self.possible_match_threshold:
            decision = "POSSIBLE_MATCH"
            action = "NEEDS_INVESTIGATION"
        else:
            decision = "NOT_MATCH"
            action = "KEEP_SEPARATE"

        # Generate explanation
        explanation = self.generate_explanation(
            decision,
            weighted_score,
            string_scores,
            gnn_score,
            graph_signals,
            llm_analysis
        )

        return EntityResolutionDecision(
            decision=decision,
            confidence=weighted_score,
            action=action,
            explanation=explanation,
            contributing_signals={
                "string_similarity": {
                    "name": string_scores.name_similarity,
                    "address": string_scores.address_similarity,
                    "identifier": string_scores.identifier_match_score
                },
                "gnn_match": gnn_score,
                "graph_patterns": {
                    "shared_directors": graph_signals.shared_directors_score,
                    "shared_addresses": graph_signals.shared_addresses_score,
                    "num_shared_directors": graph_signals.num_shared_directors
                },
                "llm_analysis": {
                    "is_match": llm_analysis.is_match,
                    "confidence": llm_analysis.confidence,
                    "reasoning": llm_analysis.reasoning
                }
            }
        )

    def generate_explanation(
        self,
        decision: str,
        score: float,
        string_scores: StringSimilarityScores,
        gnn_score: float,
        graph_signals: GraphSignals,
        llm_analysis: DisambiguationResult
    ) -> str:
        """
        Generate human-readable explanation for decision

        Args:
            decision: Decision type
            score: Overall confidence score
            string_scores: String similarity scores
            gnn_score: GNN score
            graph_signals: Graph pattern signals
            llm_analysis: LLM analysis

        Returns:
            Explanation string
        """
        explanation_parts = []

        explanation_parts.append(
            f"Entity resolution decision: {decision} (confidence: {score:.1%})"
        )

        # String similarity contribution
        if string_scores.name_similarity > 0.8:
            explanation_parts.append(
                f"✓ Strong name similarity ({string_scores.name_similarity:.1%})"
            )
        elif string_scores.name_similarity > 0.5:
            explanation_parts.append(
                f"• Moderate name similarity ({string_scores.name_similarity:.1%})"
            )

        if string_scores.identifier_match_score > 0:
            explanation_parts.append(
                f"✓ Matching identifiers detected (confidence: {string_scores.identifier_match_score:.1%})"
            )

        if string_scores.address_similarity > 0.7:
            explanation_parts.append(
                f"✓ Address similarity ({string_scores.address_similarity:.1%})"
            )

        # GNN contribution
        if gnn_score > 0.7:
            explanation_parts.append(
                f"✓ Graph neural network indicates high match probability ({gnn_score:.1%})"
            )
        elif gnn_score > 0.4:
            explanation_parts.append(
                f"• Graph neural network indicates moderate match probability ({gnn_score:.1%})"
            )

        # Graph patterns
        if graph_signals.num_shared_directors > 0:
            explanation_parts.append(
                f"✓ Entities share {graph_signals.num_shared_directors} director(s)"
            )

        if graph_signals.shared_addresses_score > 0:
            explanation_parts.append(
                f"✓ Entities share registered address"
            )

        # LLM reasoning
        if llm_analysis.reasoning:
            explanation_parts.append(f"\nLLM Analysis: {llm_analysis.reasoning}")

        if llm_analysis.supporting_factors:
            explanation_parts.append(
                "\nSupporting factors: " + ", ".join(llm_analysis.supporting_factors)
            )

        if llm_analysis.contradicting_factors:
            explanation_parts.append(
                "\nContradicting factors: " + ", ".join(llm_analysis.contradicting_factors)
            )

        return "\n".join(explanation_parts)


class MatchCache:
    """
    Cache for entity resolution decisions
    """

    def __init__(self):
        """Initialize match cache"""
        self.cache: Dict[Tuple[str, str], EntityResolutionDecision] = {}

    def get(self, entity1_id: str, entity2_id: str) -> Optional[EntityResolutionDecision]:
        """
        Get cached decision

        Args:
            entity1_id: First entity ID
            entity2_id: Second entity ID

        Returns:
            Cached decision or None
        """
        key = self._make_key(entity1_id, entity2_id)
        return self.cache.get(key)

    def put(
        self,
        entity1_id: str,
        entity2_id: str,
        decision: EntityResolutionDecision
    ) -> None:
        """
        Cache decision

        Args:
            entity1_id: First entity ID
            entity2_id: Second entity ID
            decision: Decision to cache
        """
        key = self._make_key(entity1_id, entity2_id)
        self.cache[key] = decision

    def _make_key(self, entity1_id: str, entity2_id: str) -> Tuple[str, str]:
        """
        Create cache key (order-independent)

        Args:
            entity1_id: First entity ID
            entity2_id: Second entity ID

        Returns:
            Cache key tuple
        """
        return tuple(sorted([entity1_id, entity2_id]))

    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()

    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


class GraphPatternAnalyzer:
    """
    Analyze graph patterns between entities
    """

    def __init__(self, graph_db):
        """
        Initialize graph pattern analyzer

        Args:
            graph_db: Neo4j database connection
        """
        self.graph_db = graph_db

    async def analyze_entity_pair(
        self,
        entity1_id: str,
        entity2_id: str
    ) -> GraphSignals:
        """
        Analyze graph patterns between two entities

        Args:
            entity1_id: First entity ID
            entity2_id: Second entity ID

        Returns:
            GraphSignals object
        """
        # Find shared directors
        shared_directors = await self._find_shared_directors(entity1_id, entity2_id)

        # Find shared addresses
        shared_addresses = await self._find_shared_addresses(entity1_id, entity2_id)

        # Find common owners
        common_owners = await self._find_common_owners(entity1_id, entity2_id)

        # Compute scores
        shared_directors_score = min(len(shared_directors) / 3.0, 1.0)
        shared_addresses_score = float(len(shared_addresses) > 0)
        common_owners_score = min(len(common_owners) / 2.0, 1.0)

        return GraphSignals(
            shared_directors=shared_directors,
            num_shared_directors=len(shared_directors),
            shared_directors_score=shared_directors_score,
            shared_addresses=shared_addresses,
            shared_addresses_score=shared_addresses_score,
            common_owners=common_owners,
            common_owners_score=common_owners_score
        )

    async def _find_shared_directors(
        self,
        entity1_id: str,
        entity2_id: str
    ) -> List[Entity]:
        """Find directors shared between two entities"""
        query = """
        MATCH (e1:Entity {entity_id: $entity1_id})<-[r1:RELATIONSHIP]-(director:Entity)
        MATCH (e2:Entity {entity_id: $entity2_id})<-[r2:RELATIONSHIP]-(director)
        WHERE r1.relationship_type = 'DIRECTOR_OF'
        AND r2.relationship_type = 'DIRECTOR_OF'
        AND director.entity_type = 'INDIVIDUAL'
        RETURN DISTINCT director
        """

        results = self.graph_db.execute_query(
            query,
            {"entity1_id": entity1_id, "entity2_id": entity2_id}
        )

        # Convert results to Entity objects
        directors = []
        # Parse results here
        return directors

    async def _find_shared_addresses(
        self,
        entity1_id: str,
        entity2_id: str
    ) -> List[str]:
        """Find shared addresses between two entities"""
        query = """
        MATCH (e1:Entity {entity_id: $entity1_id})
        MATCH (e2:Entity {entity_id: $entity2_id})
        WITH e1, e2,
             [addr IN e1.addresses WHERE addr IN e2.addresses] AS shared_addrs
        WHERE size(shared_addrs) > 0
        RETURN shared_addrs
        """

        results = self.graph_db.execute_query(
            query,
            {"entity1_id": entity1_id, "entity2_id": entity2_id}
        )

        if results:
            return results[0].get("shared_addrs", [])

        return []

    async def _find_common_owners(
        self,
        entity1_id: str,
        entity2_id: str
    ) -> List[Entity]:
        """Find common owners of two entities"""
        query = """
        MATCH (e1:Entity {entity_id: $entity1_id})<-[r1:RELATIONSHIP]-(owner:Entity)
        MATCH (e2:Entity {entity_id: $entity2_id})<-[r2:RELATIONSHIP]-(owner)
        WHERE r1.relationship_type IN ['OWNS', 'CONTROLS']
        AND r2.relationship_type IN ['OWNS', 'CONTROLS']
        RETURN DISTINCT owner
        """

        results = self.graph_db.execute_query(
            query,
            {"entity1_id": entity1_id, "entity2_id": entity2_id}
        )

        # Convert results to Entity objects
        owners = []
        # Parse results here
        return owners
