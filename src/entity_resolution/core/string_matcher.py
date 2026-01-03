"""
String similarity matching for entity resolution
"""
import re
from typing import Dict, List
from difflib import SequenceMatcher

try:
    import jellyfish
    JELLYFISH_AVAILABLE = True
except ImportError:
    JELLYFISH_AVAILABLE = False
    jellyfish = None

from src.entity_resolution.core.entities import Entity, StringSimilarityScores


class StringSimilarityMatcher:
    """
    Traditional string similarity matching using multiple algorithms
    """

    def __init__(self):
        """Initialize string matcher"""
        self.has_jellyfish = JELLYFISH_AVAILABLE

    def compute_similarity(
        self,
        entity1: Entity,
        entity2: Entity
    ) -> StringSimilarityScores:
        """
        Compute various string similarity metrics between two entities

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            StringSimilarityScores object
        """
        # Name similarity
        name_sim = self.compute_name_similarity(entity1.name, entity2.name)

        # Address similarity
        addr_sim = self.compute_address_similarity(
            entity1.addresses,
            entity2.addresses
        )

        # Identifier exact match
        identifier_match = self.compute_identifier_match(
            entity1.identifiers,
            entity2.identifiers
        )

        return StringSimilarityScores(
            name_similarity=name_sim,
            address_similarity=addr_sim,
            identifier_match_score=identifier_match
        )

    def compute_name_similarity(self, name1: str, name2: str) -> float:
        """
        Compute name similarity using multiple algorithms

        Args:
            name1: First name
            name2: Second name

        Returns:
            Similarity score (0-1)
        """
        if not name1 or not name2:
            return 0.0

        # Normalize names for comparison
        name1_norm = name1.lower().strip()
        name2_norm = name2.lower().strip()

        # Exact match
        if name1_norm == name2_norm:
            return 1.0

        scores = []

        # 1. Jaro-Winkler similarity (if jellyfish available)
        if self.has_jellyfish:
            jaro_winkler = jellyfish.jaro_winkler_similarity(name1_norm, name2_norm)
            scores.append(jaro_winkler)
        else:
            # Fallback to SequenceMatcher
            scores.append(SequenceMatcher(None, name1_norm, name2_norm).ratio())

        # 2. Levenshtein distance (normalized)
        if self.has_jellyfish:
            lev_distance = jellyfish.levenshtein_distance(name1_norm, name2_norm)
            max_len = max(len(name1_norm), len(name2_norm))
            lev_similarity = 1 - (lev_distance / max_len) if max_len > 0 else 0
            scores.append(lev_similarity)

        # 3. Token set ratio (handles word order differences)
        token_set_score = self._token_set_ratio(name1_norm, name2_norm)
        scores.append(token_set_score)

        # 4. Token sort ratio
        token_sort_score = self._token_sort_ratio(name1_norm, name2_norm)
        scores.append(token_sort_score)

        # Weighted average
        if len(scores) == 4:
            similarity = (
                0.3 * scores[0] +  # Jaro-Winkler
                0.2 * scores[1] +  # Levenshtein
                0.3 * scores[2] +  # Token set
                0.2 * scores[3]    # Token sort
            )
        else:
            similarity = sum(scores) / len(scores)

        return min(1.0, max(0.0, similarity))

    def compute_address_similarity(
        self,
        addresses1: List[str],
        addresses2: List[str]
    ) -> float:
        """
        Compute address similarity

        Args:
            addresses1: First entity's addresses
            addresses2: Second entity's addresses

        Returns:
            Similarity score (0-1)
        """
        if not addresses1 or not addresses2:
            return 0.0

        # Find best matching address pair
        max_similarity = 0.0

        for addr1 in addresses1:
            for addr2 in addresses2:
                # Normalize addresses
                norm_addr1 = self._normalize_for_comparison(addr1)
                norm_addr2 = self._normalize_for_comparison(addr2)

                # Exact match
                if norm_addr1 == norm_addr2:
                    return 1.0

                # Compute similarity
                if self.has_jellyfish:
                    similarity = jellyfish.jaro_winkler_similarity(norm_addr1, norm_addr2)
                else:
                    similarity = SequenceMatcher(None, norm_addr1, norm_addr2).ratio()

                max_similarity = max(max_similarity, similarity)

        return max_similarity

    def compute_identifier_match(
        self,
        identifiers1: Dict[str, str],
        identifiers2: Dict[str, str]
    ) -> float:
        """
        Check for exact identifier matches

        Args:
            identifiers1: First entity's identifiers
            identifiers2: Second entity's identifiers

        Returns:
            Match score (0 or 1)
        """
        if not identifiers1 or not identifiers2:
            return 0.0

        # Priority ordered identifier types
        identifier_types = ["ein", "lei", "company_number", "duns", "ssn", "vat", "isin"]

        for id_type in identifier_types:
            if id_type in identifiers1 and id_type in identifiers2:
                # Exact match on high-confidence identifier
                if identifiers1[id_type] == identifiers2[id_type]:
                    return 1.0

        # Partial match: any common identifier
        common_ids = set(identifiers1.keys()) & set(identifiers2.keys())
        for id_type in common_ids:
            if identifiers1[id_type] == identifiers2[id_type]:
                return 0.5  # Lower confidence for non-primary identifiers

        return 0.0

    def _token_set_ratio(self, str1: str, str2: str) -> float:
        """
        Token set ratio - ignores word order and duplicates

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score (0-1)
        """
        tokens1 = set(str1.split())
        tokens2 = set(str2.split())

        if not tokens1 or not tokens2:
            return 0.0

        # Intersection over union
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union) if union else 0.0

    def _token_sort_ratio(self, str1: str, str2: str) -> float:
        """
        Token sort ratio - sorts tokens before comparison

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score (0-1)
        """
        # Sort tokens alphabetically
        sorted_str1 = ' '.join(sorted(str1.split()))
        sorted_str2 = ' '.join(sorted(str2.split()))

        # Compute similarity on sorted strings
        return SequenceMatcher(None, sorted_str1, sorted_str2).ratio()

    def _normalize_for_comparison(self, text: str) -> str:
        """
        Normalize text for comparison

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Convert to lowercase
        normalized = text.lower()

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        return normalized.strip()


class FuzzyMatcher:
    """
    Fuzzy matching utilities for entity names
    """

    @staticmethod
    def is_potential_match(name1: str, name2: str, threshold: float = 0.8) -> bool:
        """
        Quick check if two names are potential matches

        Args:
            name1: First name
            name2: Second name
            threshold: Similarity threshold (0-1)

        Returns:
            True if potential match
        """
        if not name1 or not name2:
            return False

        # Quick exact match check
        if name1.lower().strip() == name2.lower().strip():
            return True

        # Quick similarity check
        ratio = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
        return ratio >= threshold

    @staticmethod
    def get_common_substrings(str1: str, str2: str, min_length: int = 3) -> List[str]:
        """
        Find common substrings between two strings

        Args:
            str1: First string
            str2: Second string
            min_length: Minimum substring length

        Returns:
            List of common substrings
        """
        common = []
        matcher = SequenceMatcher(None, str1.lower(), str2.lower())

        for match in matcher.get_matching_blocks():
            if match.size >= min_length:
                common.append(str1[match.a:match.a + match.size])

        return common

    @staticmethod
    def longest_common_substring(str1: str, str2: str) -> str:
        """
        Find longest common substring

        Args:
            str1: First string
            str2: Second string

        Returns:
            Longest common substring
        """
        matcher = SequenceMatcher(None, str1.lower(), str2.lower())
        match = matcher.find_longest_match(0, len(str1), 0, len(str2))

        if match.size > 0:
            return str1[match.a:match.a + match.size]

        return ""


class PhoneticMatcher:
    """
    Phonetic matching for names (useful for handling spelling variations)
    """

    @staticmethod
    def soundex(name: str) -> str:
        """
        Compute Soundex code for name

        Args:
            name: Name to encode

        Returns:
            Soundex code
        """
        if JELLYFISH_AVAILABLE:
            return jellyfish.soundex(name)

        # Simple fallback implementation
        if not name:
            return ""

        name = name.upper()

        # Keep first letter
        soundex_code = name[0]

        # Mapping of letters to digits
        mapping = {
            'BFPV': '1',
            'CGJKQSXZ': '2',
            'DT': '3',
            'L': '4',
            'MN': '5',
            'R': '6'
        }

        # Convert remaining letters
        for char in name[1:]:
            for key, value in mapping.items():
                if char in key:
                    if soundex_code[-1] != value:  # Avoid duplicates
                        soundex_code += value
                    break

        # Pad with zeros or truncate to 4 characters
        soundex_code = (soundex_code + '000')[:4]

        return soundex_code

    @staticmethod
    def metaphone(name: str) -> str:
        """
        Compute Metaphone code for name

        Args:
            name: Name to encode

        Returns:
            Metaphone code
        """
        if JELLYFISH_AVAILABLE:
            return jellyfish.metaphone(name)

        # Fallback to soundex
        return PhoneticMatcher.soundex(name)

    @staticmethod
    def phonetic_match(name1: str, name2: str) -> bool:
        """
        Check if two names match phonetically

        Args:
            name1: First name
            name2: Second name

        Returns:
            True if phonetic match
        """
        soundex1 = PhoneticMatcher.soundex(name1)
        soundex2 = PhoneticMatcher.soundex(name2)

        return soundex1 == soundex2
