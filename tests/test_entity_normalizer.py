"""
Tests for entity normalizer
"""
import pytest
from src.entity_resolution.core.entity_normalizer import EntityNormalizer, PhoneNumberNormalizer
from src.entity_resolution.core.entities import Entity, EntityType


class TestEntityNormalizer:
    """Test cases for EntityNormalizer"""

    def setup_method(self):
        """Setup test fixtures"""
        self.normalizer = EntityNormalizer()

    def test_normalize_company_name(self):
        """Test company name normalization"""
        # Test legal suffix removal
        assert self.normalizer.normalize_name("ABC Corporation") == "Abc"
        assert self.normalizer.normalize_name("XYZ Inc.") == "Xyz"
        assert self.normalizer.normalize_name("DEF LLC") == "Def"

        # Test punctuation removal
        assert self.normalizer.normalize_name("A&B Company") == "A B"

        # Test whitespace normalization
        assert self.normalizer.normalize_name("Test   Company   Inc") == "Test"

    def test_normalize_address(self):
        """Test address normalization"""
        addr = "123 Main Street, New York, NY"
        normalized = self.normalizer.normalize_address(addr)

        # Should be uppercase
        assert normalized.isupper()

        # Should have standardized street suffix
        assert "ST" in normalized

        # Should have state abbreviation
        assert "NY" in normalized

    def test_normalize_identifiers(self):
        """Test identifier normalization"""
        identifiers = {
            "ein": "12-3456789",
            "lei": "1234567890ABCDEFGHIJ",
            "company_number": "ABC 123"
        }

        normalized = self.normalizer.normalize_identifiers(identifiers)

        # EIN should have no hyphens
        assert normalized["ein"] == "123456789"

        # LEI should be uppercase
        assert normalized["lei"] == "1234567890ABCDEFGHIJ"

        # Company number should have no spaces
        assert normalized["company_number"] == "ABC123"

    def test_normalize_entity(self):
        """Test full entity normalization"""
        entity = Entity(
            name="Test Company, Inc.",
            entity_type=EntityType.COMPANY,
            addresses=["123 Main Street"],
            identifiers={"ein": "12-3456789"}
        )

        normalized = self.normalizer.normalize(entity)

        assert "Inc" not in normalized.name
        assert len(normalized.addresses) == 1
        assert normalized.identifiers["ein"] == "123456789"


class TestPhoneNumberNormalizer:
    """Test cases for PhoneNumberNormalizer"""

    def test_normalize_phone(self):
        """Test phone number normalization"""
        # Test various formats
        assert PhoneNumberNormalizer.normalize("(555) 123-4567") == "5551234567"
        assert PhoneNumberNormalizer.normalize("555.123.4567") == "5551234567"
        assert PhoneNumberNormalizer.normalize("+1-555-123-4567") == "5551234567"

        # Test empty input
        assert PhoneNumberNormalizer.normalize("") == ""
