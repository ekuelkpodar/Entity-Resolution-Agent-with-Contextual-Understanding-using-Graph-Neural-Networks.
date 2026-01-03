"""
Entity extraction from documents using NER and LLM
"""
import json
import re
from typing import List, Optional
from datetime import datetime

from src.entity_resolution.core.entities import Entity, EntityType, Document
from src.entity_resolution.core.entity_normalizer import EntityNormalizer


class EntityExtractor:
    """
    Extract and normalize entities from various data sources
    Uses hybrid approach: NER + LLM extraction
    """

    def __init__(self, llm_client: Optional[any] = None):
        """
        Initialize entity extractor

        Args:
            llm_client: LLM client (Anthropic Claude) for advanced extraction
        """
        self.llm_client = llm_client
        self.normalizer = EntityNormalizer()

        # Try to load spaCy model if available
        try:
            import spacy
            self.ner_model = spacy.load("en_core_web_sm")
            self.has_spacy = True
        except (ImportError, OSError):
            self.ner_model = None
            self.has_spacy = False

    async def extract_entities(self, document: Document) -> List[Entity]:
        """
        Extract entities from document using hybrid approach
        """
        entities = []

        # Step 1: NER extraction for structured entities (if spaCy available)
        if self.has_spacy and self.ner_model:
            ner_entities = self.extract_with_ner(document.text)
            entities.extend(ner_entities)

        # Step 2: LLM-based extraction for complex entities (if LLM available)
        if self.llm_client:
            llm_entities = await self.extract_with_llm(document.text)
            entities.extend(llm_entities)

        # Step 3: Rule-based extraction as fallback
        if not entities:
            rule_entities = self.extract_with_rules(document.text)
            entities.extend(rule_entities)

        # Step 4: Deduplicate entities
        unique_entities = self.deduplicate_entities(entities)

        # Step 5: Normalize entity attributes
        normalized_entities = [
            self.normalizer.normalize(entity)
            for entity in unique_entities
        ]

        return normalized_entities

    def extract_with_ner(self, text: str) -> List[Entity]:
        """
        Use spaCy NER for basic entity extraction
        """
        if not self.ner_model:
            return []

        doc = self.ner_model(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "GPE"]:
                entity = Entity(
                    name=ent.text,
                    original_name=ent.text,
                    entity_type=self._map_entity_type(ent.label_),
                    confidence=0.8,  # Default NER confidence
                    data_sources=["spacy_ner"]
                )
                entities.append(entity)

        return entities

    async def extract_with_llm(self, text: str) -> List[Entity]:
        """
        Use LLM for complex entity extraction
        """
        if not self.llm_client:
            return []

        prompt = f"""Extract all business entities, individuals, and organizations from the following text.
For each entity, identify:
- Entity name
- Entity type (COMPANY, INDIVIDUAL, GOVERNMENT, NGO)
- Roles (if mentioned): director, shareholder, beneficial owner, nominee
- Ownership percentages (if mentioned)
- Addresses (if mentioned)
- Relationships to other entities (if mentioned)

Text: {text}

Return a JSON array of entities with the structure:
{{
    "entities": [
        {{
            "name": "string",
            "type": "COMPANY|INDIVIDUAL|GOVERNMENT|NGO",
            "roles": ["string"],
            "ownership_pct": float or null,
            "addresses": ["string"],
            "jurisdiction": "string" or null,
            "incorporation_date": "YYYY-MM-DD" or null,
            "dissolved_date": "YYYY-MM-DD" or null,
            "identifiers": {{
                "ein": "string" or null,
                "company_number": "string" or null
            }}
        }}
    ]
}}
"""

        try:
            # Call LLM (Anthropic Claude)
            response = await self.llm_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=3000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            content = response.content[0].text
            entities_data = json.loads(content)

            entities = []
            for entity_dict in entities_data.get("entities", []):
                entity = self._parse_llm_entity(entity_dict)
                entities.append(entity)

            return entities

        except Exception as e:
            print(f"LLM extraction error: {e}")
            return []

    def extract_with_rules(self, text: str) -> List[Entity]:
        """
        Rule-based entity extraction as fallback
        """
        entities = []

        # Extract company names (simple heuristic: words followed by Inc, LLC, Corp, etc.)
        company_pattern = r'\b([A-Z][A-Za-z\s&]+(?:Inc\.?|LLC|Corp\.?|Corporation|Ltd\.?|Limited|GmbH|SA|AG))\b'
        for match in re.finditer(company_pattern, text):
            entity = Entity(
                name=match.group(1),
                original_name=match.group(1),
                entity_type=EntityType.COMPANY,
                confidence=0.6,  # Lower confidence for rule-based
                data_sources=["rule_based"]
            )
            entities.append(entity)

        # Extract potential individual names (capitalized consecutive words)
        name_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b'
        for match in re.finditer(name_pattern, text):
            # Filter out common false positives
            name = match.group(1)
            if not self._is_false_positive_name(name):
                entity = Entity(
                    name=name,
                    original_name=name,
                    entity_type=EntityType.INDIVIDUAL,
                    confidence=0.5,
                    data_sources=["rule_based"]
                )
                entities.append(entity)

        # Extract email addresses and associate with entities
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)

        # Extract phone numbers
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        phones = re.findall(phone_pattern, text)

        # Extract addresses (simple version)
        address_pattern = r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)\b'
        addresses = re.findall(address_pattern, text)

        # Extract EIN numbers
        ein_pattern = r'\b\d{2}-\d{7}\b'
        eins = re.findall(ein_pattern, text)

        # Add metadata to entities if found nearby
        for entity in entities:
            # Add emails found in same context
            entity.email_addresses = emails[:1] if emails else []
            # Add phones found in same context
            entity.phone_numbers = phones[:1] if phones else []
            # Add addresses found in same context
            entity.addresses = addresses[:1] if addresses else []
            # Add EINs if found
            if eins:
                entity.identifiers["ein"] = eins[0]

        return entities

    def deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove duplicate entities based on name similarity
        """
        if not entities:
            return []

        unique_entities = []
        seen_names = set()

        for entity in entities:
            # Normalize name for comparison
            norm_name = entity.name.lower().strip()

            if norm_name not in seen_names:
                seen_names.add(norm_name)
                unique_entities.append(entity)
            else:
                # If duplicate, merge metadata from higher confidence entity
                for existing in unique_entities:
                    if existing.name.lower().strip() == norm_name:
                        if entity.confidence > existing.confidence:
                            # Keep higher confidence entity
                            existing.confidence = entity.confidence
                        # Merge data sources
                        existing.data_sources = list(set(existing.data_sources + entity.data_sources))
                        break

        return unique_entities

    def _map_entity_type(self, ner_label: str) -> EntityType:
        """Map NER label to EntityType"""
        mapping = {
            "ORG": EntityType.COMPANY,
            "PERSON": EntityType.INDIVIDUAL,
            "GPE": EntityType.GOVERNMENT,
        }
        return mapping.get(ner_label, EntityType.COMPANY)

    def _parse_llm_entity(self, entity_dict: dict) -> Entity:
        """Parse LLM entity dict to Entity object"""
        entity_type = EntityType[entity_dict.get("type", "COMPANY")]

        # Parse dates
        incorporation_date = None
        if entity_dict.get("incorporation_date"):
            try:
                incorporation_date = datetime.fromisoformat(entity_dict["incorporation_date"])
            except (ValueError, TypeError):
                pass

        dissolution_date = None
        if entity_dict.get("dissolved_date"):
            try:
                dissolution_date = datetime.fromisoformat(entity_dict["dissolved_date"])
            except (ValueError, TypeError):
                pass

        entity = Entity(
            name=entity_dict.get("name", ""),
            original_name=entity_dict.get("name", ""),
            entity_type=entity_type,
            roles=entity_dict.get("roles", []),
            addresses=entity_dict.get("addresses", []),
            jurisdiction=entity_dict.get("jurisdiction"),
            incorporation_date=incorporation_date,
            dissolution_date=dissolution_date,
            identifiers=entity_dict.get("identifiers", {}),
            confidence=0.9,  # High confidence for LLM extraction
            data_sources=["llm_extraction"]
        )

        # Add ownership percentage if mentioned
        if entity_dict.get("ownership_pct"):
            entity.roles.append(f"owner_{entity_dict['ownership_pct']}%")

        return entity

    def _is_false_positive_name(self, name: str) -> bool:
        """Filter out common false positive names"""
        false_positives = {
            "The Company", "This Agreement", "United States",
            "New York", "Los Angeles", "San Francisco",
            "Chief Executive", "Board Members"
        }
        return name in false_positives


class RelationshipExtractor:
    """Extract relationships between entities from text"""

    def __init__(self, llm_client: Optional[any] = None):
        self.llm_client = llm_client

    async def extract_relationships(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[dict]:
        """
        Extract relationships between identified entities
        """
        if not self.llm_client or not entities:
            return []

        entity_names = [e.name for e in entities]

        prompt = f"""Analyze the following text and identify relationships between these entities:

Entities: {', '.join(entity_names)}

Text: {text}

Identify relationships such as:
- Ownership (who owns whom, what percentage)
- Control (who controls whom)
- Directorship (who is a director of what)
- Shareholding (who holds shares in what)
- Beneficial ownership (who is the ultimate beneficiary)

Return a JSON array:
{{
    "relationships": [
        {{
            "source_entity": "entity name",
            "target_entity": "entity name",
            "relationship_type": "OWNS|CONTROLS|DIRECTOR_OF|SHAREHOLDER_OF|UBO_OF",
            "properties": {{
                "ownership_pct": float or null,
                "start_date": "YYYY-MM-DD" or null
            }},
            "confidence": float (0-1)
        }}
    ]
}}
"""

        try:
            response = await self.llm_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text
            relationships_data = json.loads(content)
            return relationships_data.get("relationships", [])

        except Exception as e:
            print(f"Relationship extraction error: {e}")
            return []
