"""
Contextual reasoning engine using LLM for entity resolution
"""
import json
from typing import Any, Dict, List, Optional

from src.entity_resolution.core.entities import (
    Entity,
    ContextualAnalysis,
    DisambiguationResult
)
from config.settings import settings


class ContextualReasoningEngine:
    """
    LLM-based contextual reasoning for entity resolution
    """

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize contextual reasoning engine

        Args:
            llm_client: LLM client (Anthropic Claude)
        """
        self.llm_client = llm_client

    async def analyze_entity_context(
        self,
        entity: Entity,
        graph_context: Optional[Dict[str, Any]] = None
    ) -> ContextualAnalysis:
        """
        Analyze entity using graph context and LLM reasoning

        Args:
            entity: Entity to analyze
            graph_context: Graph neighborhood context

        Returns:
            ContextualAnalysis object
        """
        if not self.llm_client:
            return ContextualAnalysis(
                profile_summary="LLM not available",
                entity_classification="UNKNOWN",
                classification_confidence=0.0
            )

        # Build context description
        context_description = self._build_context_description(entity, graph_context)

        # LLM analysis
        analysis = await self._llm_analyze_entity(context_description)

        return analysis

    async def disambiguate_entities(
        self,
        entity1: Entity,
        entity2: Entity,
        match_confidence: float,
        context1: Optional[Dict[str, Any]] = None,
        context2: Optional[Dict[str, Any]] = None
    ) -> DisambiguationResult:
        """
        Use LLM to disambiguate potentially matching entities

        Args:
            entity1: First entity
            entity2: Second entity
            match_confidence: GNN-based match confidence
            context1: Graph context for entity1
            context2: Graph context for entity2

        Returns:
            DisambiguationResult
        """
        if not self.llm_client:
            return DisambiguationResult(
                is_match=False,
                confidence=0.0,
                reasoning="LLM not available",
                recommendation="NEEDS_INVESTIGATION"
            )

        prompt = self._build_disambiguation_prompt(
            entity1,
            entity2,
            match_confidence,
            context1,
            context2
        )

        try:
            response = await self.llm_client.messages.create(
                model=settings.default_llm_model,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text

            # Extract JSON from response
            result_data = self._extract_json(content)

            return DisambiguationResult.from_dict(result_data)

        except Exception as e:
            print(f"LLM disambiguation error: {e}")
            return DisambiguationResult(
                is_match=False,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                recommendation="NEEDS_INVESTIGATION"
            )

    def _build_context_description(
        self,
        entity: Entity,
        graph_context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build textual description of entity's context

        Args:
            entity: Entity to describe
            graph_context: Graph neighborhood information

        Returns:
            Context description string
        """
        description_parts = []

        # Entity basic info
        description_parts.append(f"Entity: {entity.name}")
        description_parts.append(f"Type: {entity.entity_type.value}")

        if entity.jurisdiction:
            description_parts.append(f"Jurisdiction: {entity.jurisdiction}")

        if entity.incorporation_date:
            description_parts.append(f"Incorporated: {entity.incorporation_date.date()}")

        if entity.addresses:
            description_parts.append(f"Address: {entity.addresses[0]}")

        # Identifiers
        if entity.identifiers:
            id_str = ", ".join([f"{k}: {v}" for k, v in entity.identifiers.items()])
            description_parts.append(f"Identifiers: {id_str}")

        # Graph context
        if graph_context:
            if graph_context.get("owners"):
                owners = ", ".join(graph_context["owners"])
                description_parts.append(f"Owned by: {owners}")

            if graph_context.get("subsidiaries"):
                subs = ", ".join(graph_context["subsidiaries"])
                description_parts.append(f"Subsidiaries: {subs}")

            if graph_context.get("directors"):
                dirs = ", ".join(graph_context["directors"])
                description_parts.append(f"Directors: {dirs}")

            if graph_context.get("shared_addresses"):
                shared = ", ".join(graph_context["shared_addresses"])
                description_parts.append(f"Shares address with: {shared}")

        return "\n".join(description_parts)

    async def _llm_analyze_entity(self, context: str) -> ContextualAnalysis:
        """
        Use LLM to analyze entity context

        Args:
            context: Entity context description

        Returns:
            ContextualAnalysis
        """
        prompt = f"""You are an expert in corporate intelligence and anti-money laundering analysis.
Analyze the following entity and its business relationships:

{context}

Based on this information, provide:

1. Entity Profile Summary (2-3 sentences)
   - What type of entity is this?
   - What is its primary purpose/business?
   - Any notable characteristics?

2. Risk Indicators (if any)
   - Shell company indicators (no employees, nominee directors, tax haven jurisdiction)
   - Ownership opacity (complex structures, offshore entities)
   - Network red flags (connections to sanctioned entities, PEPs)
   - Temporal anomalies (rapid creation/dissolution, recent restructuring)

3. Beneficial Ownership Assessment
   - Can you identify the ultimate beneficial owner(s)?
   - Is the ownership structure transparent or opaque?
   - Any concerns about nominee arrangements?

4. Entity Purpose Classification
   - Operating company (actual business operations)
   - Holding company (asset holding, no operations)
   - Shell company (minimal substance, potential vehicle for illicit activity)
   - Special purpose vehicle (legitimate financial structuring)
   - Unknown/Insufficient information

5. Contextual Notes
   - Any additional observations
   - Information gaps
   - Recommended further investigation

Return your analysis in JSON format:
{{
    "profile_summary": "string",
    "risk_indicators": [
        {{
            "indicator_type": "string",
            "description": "string",
            "severity": "LOW|MEDIUM|HIGH"
        }}
    ],
    "beneficial_ownership": {{
        "ultimate_owners": ["string"],
        "ownership_transparency": "TRANSPARENT|OPAQUE|UNKNOWN",
        "nominee_concerns": true/false,
        "explanation": "string"
    }},
    "entity_classification": "OPERATING|HOLDING|SHELL|SPV|UNKNOWN",
    "classification_confidence": 0.0-1.0,
    "contextual_notes": "string",
    "recommended_actions": ["string"]
}}
"""

        try:
            response = await self.llm_client.messages.create(
                model=settings.default_llm_model,
                max_tokens=2000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text
            analysis_data = self._extract_json(content)

            return ContextualAnalysis.from_dict(analysis_data)

        except Exception as e:
            print(f"LLM analysis error: {e}")
            return ContextualAnalysis(
                profile_summary="Analysis failed",
                entity_classification="UNKNOWN",
                classification_confidence=0.0,
                contextual_notes=f"Error: {str(e)}"
            )

    def _build_disambiguation_prompt(
        self,
        entity1: Entity,
        entity2: Entity,
        match_confidence: float,
        context1: Optional[Dict[str, Any]],
        context2: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for entity disambiguation

        Args:
            entity1: First entity
            entity2: Second entity
            match_confidence: GNN match confidence
            context1: Context for entity1
            context2: Context for entity2

        Returns:
            Disambiguation prompt
        """
        prompt = f"""You are an expert in entity resolution for financial crime compliance.
Determine whether these two entities are the same entity (a match) or different entities.

Entity 1:
Name: {entity1.name}
Type: {entity1.entity_type.value}
Jurisdiction: {entity1.jurisdiction or 'Unknown'}
Addresses: {', '.join(entity1.addresses) if entity1.addresses else 'None'}
Incorporation Date: {entity1.incorporation_date.date() if entity1.incorporation_date else 'Unknown'}
Identifiers: {', '.join([f"{k}: {v}" for k, v in entity1.identifiers.items()]) if entity1.identifiers else 'None'}

Entity 2:
Name: {entity2.name}
Type: {entity2.entity_type.value}
Jurisdiction: {entity2.jurisdiction or 'Unknown'}
Addresses: {', '.join(entity2.addresses) if entity2.addresses else 'None'}
Incorporation Date: {entity2.incorporation_date.date() if entity2.incorporation_date else 'Unknown'}
Identifiers: {', '.join([f"{k}: {v}" for k, v in entity2.identifiers.items()]) if entity2.identifiers else 'None'}

Current GNN-based match confidence: {match_confidence:.2%}

Consider:
1. Name similarity (accounting for legal suffixes, abbreviations)
2. Jurisdiction match
3. Address overlap
4. Temporal consistency (incorporation dates)
5. Identifier matches (if any)
6. Business context alignment
7. Network context (do they have the same owners/directors?)

Provide your assessment in JSON format:
{{
    "is_match": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of your decision",
    "supporting_factors": ["factor 1", "factor 2"],
    "contradicting_factors": ["factor 1", "factor 2"],
    "recommendation": "MERGE|KEEP_SEPARATE|NEEDS_INVESTIGATION"
}}
"""

        return prompt

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response

        Args:
            text: LLM response text

        Returns:
            Parsed JSON dictionary
        """
        # Try to find JSON in the response
        import re

        # Look for JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)

        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # If direct parsing fails, try to clean and parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Return minimal valid response
            return {
                "profile_summary": "Unable to parse LLM response",
                "entity_classification": "UNKNOWN",
                "classification_confidence": 0.0
            }


class RiskAssessment:
    """
    Risk assessment utilities for entities
    """

    # Tax haven jurisdictions
    TAX_HAVENS = {
        "Cayman Islands", "British Virgin Islands", "Bermuda", "Panama",
        "Bahamas", "Jersey", "Guernsey", "Isle of Man", "Seychelles",
        "Mauritius", "Luxembourg", "Switzerland", "Singapore", "Hong Kong",
        "Malta", "Cyprus", "Liechtenstein", "Monaco", "Andorra"
    }

    # High-risk jurisdictions (for AML purposes)
    HIGH_RISK_JURISDICTIONS = {
        "North Korea", "Iran", "Syria", "Myanmar", "Afghanistan"
    }

    @classmethod
    def is_tax_haven(cls, jurisdiction: Optional[str]) -> bool:
        """
        Check if jurisdiction is a tax haven

        Args:
            jurisdiction: Jurisdiction name

        Returns:
            True if tax haven
        """
        if not jurisdiction:
            return False

        return jurisdiction in cls.TAX_HAVENS

    @classmethod
    def is_high_risk_jurisdiction(cls, jurisdiction: Optional[str]) -> bool:
        """
        Check if jurisdiction is high-risk for AML

        Args:
            jurisdiction: Jurisdiction name

        Returns:
            True if high-risk
        """
        if not jurisdiction:
            return False

        return jurisdiction in cls.HIGH_RISK_JURISDICTIONS

    @classmethod
    def calculate_jurisdiction_risk_score(cls, jurisdiction: Optional[str]) -> float:
        """
        Calculate risk score for jurisdiction

        Args:
            jurisdiction: Jurisdiction name

        Returns:
            Risk score (0-1, higher is riskier)
        """
        if not jurisdiction:
            return 0.5  # Unknown = medium risk

        if cls.is_high_risk_jurisdiction(jurisdiction):
            return 1.0  # High risk

        if cls.is_tax_haven(jurisdiction):
            return 0.7  # Elevated risk

        return 0.1  # Low risk

    @staticmethod
    def calculate_entity_age_risk(entity: Entity) -> float:
        """
        Calculate risk based on entity age

        Args:
            entity: Entity to assess

        Returns:
            Risk score (0-1, higher is riskier)
        """
        from datetime import datetime

        if not entity.incorporation_date:
            return 0.5  # Unknown = medium risk

        age_days = (datetime.now() - entity.incorporation_date).days

        # Very new entities are riskier
        if age_days < 90:  # Less than 3 months
            return 0.8
        elif age_days < 365:  # Less than 1 year
            return 0.5
        elif age_days < 365 * 3:  # Less than 3 years
            return 0.3
        else:
            return 0.1  # Well-established

    @staticmethod
    def has_shell_company_indicators(entity: Entity) -> List[str]:
        """
        Check for shell company indicators

        Args:
            entity: Entity to check

        Returns:
            List of shell company indicators found
        """
        indicators = []

        # No contact information
        if not entity.phone_numbers and not entity.email_addresses:
            indicators.append("No contact information")

        # No website
        if not entity.websites:
            indicators.append("No website")

        # Minimal addresses or PO Box
        if not entity.addresses:
            indicators.append("No registered address")
        elif any("P.O. BOX" in addr.upper() or "PO BOX" in addr.upper() for addr in entity.addresses):
            indicators.append("PO Box address")

        # Tax haven jurisdiction
        if RiskAssessment.is_tax_haven(entity.jurisdiction):
            indicators.append(f"Tax haven jurisdiction: {entity.jurisdiction}")

        # Very new entity
        from datetime import datetime
        if entity.incorporation_date:
            age_days = (datetime.now() - entity.incorporation_date).days
            if age_days < 90:
                indicators.append(f"Very new entity (< 3 months)")

        return indicators
