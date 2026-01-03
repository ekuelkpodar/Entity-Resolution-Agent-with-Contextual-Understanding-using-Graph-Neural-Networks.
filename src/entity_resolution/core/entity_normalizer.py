"""
Entity normalization utilities
"""
import re
from typing import Dict, List
from src.entity_resolution.core.entities import Entity


class EntityNormalizer:
    """
    Normalize entity attributes for consistent matching
    """

    # Legal suffixes for companies
    LEGAL_SUFFIXES = [
        "LLC", "Ltd", "Limited", "Inc", "Incorporated",
        "Corp", "Corporation", "GmbH", "SA", "SRL", "BV",
        "Pte", "Pty", "AG", "NV", "PLC", "LP", "LLP",
        "Co", "Company", "Group", "Holdings"
    ]

    # Common business terms to remove
    BUSINESS_TERMS = [
        "THE", "COMPANY", "GROUP", "HOLDINGS", "INTERNATIONAL",
        "GLOBAL", "WORLDWIDE", "ENTERPRISES", "ASSOCIATES"
    ]

    # US State abbreviations
    STATE_ABBREVIATIONS = {
        "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
        "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
        "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
        "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
        "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
        "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
        "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
        "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
        "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
        "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
        "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
        "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
        "WISCONSIN": "WI", "WYOMING": "WY"
    }

    # Street suffix abbreviations
    STREET_SUFFIXES = {
        "STREET": "ST", "AVENUE": "AVE", "ROAD": "RD", "DRIVE": "DR",
        "BOULEVARD": "BLVD", "LANE": "LN", "COURT": "CT", "CIRCLE": "CIR",
        "PLACE": "PL", "TERRACE": "TER", "PARKWAY": "PKY", "HIGHWAY": "HWY"
    }

    def normalize(self, entity: Entity) -> Entity:
        """
        Apply comprehensive normalization to entity
        """
        entity.name = self.normalize_name(entity.name)
        entity.addresses = [self.normalize_address(addr) for addr in entity.addresses]
        entity.identifiers = self.normalize_identifiers(entity.identifiers)

        return entity

    def normalize_name(self, name: str) -> str:
        """
        Normalize company/person names
        """
        if not name:
            return ""

        normalized = name.strip()

        # Remove legal suffixes
        for suffix in self.LEGAL_SUFFIXES:
            # Match suffix at end of string with optional period and comma
            suffix_pattern = rf'\b{re.escape(suffix)}\b[.,]?\s*$'
            normalized = re.sub(suffix_pattern, '', normalized, flags=re.IGNORECASE)

        # Remove punctuation except spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)

        # Normalize whitespace
        normalized = ' '.join(normalized.split())

        # Remove common business terms
        words = normalized.upper().split()
        words = [w for w in words if w not in self.BUSINESS_TERMS]
        normalized = ' '.join(words)

        # Convert to title case for consistency
        normalized = normalized.strip().title()

        return normalized

    def normalize_address(self, address: str) -> str:
        """
        Normalize addresses for matching
        """
        if not address:
            return ""

        normalized = address.strip().upper()

        # Standardize street suffixes
        for full_suffix, abbrev in self.STREET_SUFFIXES.items():
            pattern = rf'\b{full_suffix}\b'
            normalized = re.sub(pattern, abbrev, normalized)

        # Standardize state names
        for full_state, abbrev in self.STATE_ABBREVIATIONS.items():
            pattern = rf'\b{full_state}\b'
            normalized = re.sub(pattern, abbrev, normalized)

        # Remove extra punctuation
        normalized = re.sub(r'[.,#]', ' ', normalized)

        # Normalize whitespace
        normalized = ' '.join(normalized.split())

        # Standardize directional prefixes
        normalized = re.sub(r'\bNORTH\b', 'N', normalized)
        normalized = re.sub(r'\bSOUTH\b', 'S', normalized)
        normalized = re.sub(r'\bEAST\b', 'E', normalized)
        normalized = re.sub(r'\bWEST\b', 'W', normalized)

        # Standardize apartment/unit/suite
        normalized = re.sub(r'\bAPARTMENT\b', 'APT', normalized)
        normalized = re.sub(r'\bUNIT\b', 'UNIT', normalized)
        normalized = re.sub(r'\bSUITE\b', 'STE', normalized)

        return normalized.strip()

    def normalize_identifiers(self, identifiers: Dict[str, str]) -> Dict[str, str]:
        """
        Normalize company/individual identifiers
        """
        normalized = {}

        if "ein" in identifiers:
            # US Employer Identification Number (format: XX-XXXXXXX)
            normalized["ein"] = re.sub(r'\D', '', identifiers["ein"])

        if "lei" in identifiers:
            # Legal Entity Identifier (20 characters)
            normalized["lei"] = identifiers["lei"].upper().strip()

        if "company_number" in identifiers:
            # Company registration number (remove spaces and special chars)
            normalized["company_number"] = re.sub(r'\s', '', identifiers["company_number"]).upper()

        if "ssn" in identifiers:
            # Social Security Number (encrypted in production)
            normalized["ssn"] = re.sub(r'\D', '', identifiers["ssn"])

        if "duns" in identifiers:
            # Dun & Bradstreet Number
            normalized["duns"] = re.sub(r'\D', '', identifiers["duns"])

        if "vat" in identifiers:
            # VAT Number
            normalized["vat"] = identifiers["vat"].upper().strip()

        if "isin" in identifiers:
            # International Securities Identification Number
            normalized["isin"] = identifiers["isin"].upper().strip()

        # Copy through any other identifiers
        for key, value in identifiers.items():
            if key not in normalized and value:
                normalized[key] = str(value).strip()

        return normalized

    def standardize_street(self, street: str) -> str:
        """Standardize street names"""
        street = street.upper()
        for full, abbrev in self.STREET_SUFFIXES.items():
            pattern = rf'\b{full}\b'
            street = re.sub(pattern, abbrev, street)
        return street

    def standardize_state(self, state: str) -> str:
        """Standardize US state names"""
        state_upper = state.upper().strip()

        # If already an abbreviation, return as-is
        if state_upper in self.STATE_ABBREVIATIONS.values():
            return state_upper

        # If full name, convert to abbreviation
        if state_upper in self.STATE_ABBREVIATIONS:
            return self.STATE_ABBREVIATIONS[state_upper]

        return state


class PhoneNumberNormalizer:
    """Normalize phone numbers for matching"""

    @staticmethod
    def normalize(phone: str) -> str:
        """
        Normalize phone number to digits only
        """
        if not phone:
            return ""

        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)

        # Remove leading 1 for US numbers (if 11 digits)
        if len(digits) == 11 and digits.startswith('1'):
            digits = digits[1:]

        return digits


class EmailNormalizer:
    """Normalize email addresses for matching"""

    @staticmethod
    def normalize(email: str) -> str:
        """
        Normalize email address
        """
        if not email:
            return ""

        # Convert to lowercase
        normalized = email.lower().strip()

        # Remove dots from Gmail addresses (gmail ignores them)
        if '@gmail.com' in normalized or '@googlemail.com' in normalized:
            local, domain = normalized.split('@')
            local = local.replace('.', '')
            normalized = f"{local}@{domain}"

        return normalized


class NameVariationGenerator:
    """Generate common name variations for matching"""

    @staticmethod
    def generate_variations(name: str) -> List[str]:
        """
        Generate common variations of entity name
        """
        variations = [name]

        # Add uppercase version
        variations.append(name.upper())

        # Add lowercase version
        variations.append(name.lower())

        # Add title case version
        variations.append(name.title())

        # Remove spaces
        variations.append(name.replace(' ', ''))

        # Replace spaces with hyphens
        variations.append(name.replace(' ', '-'))

        # Replace spaces with underscores
        variations.append(name.replace(' ', '_'))

        # Add common abbreviations
        if 'COMPANY' in name.upper():
            variations.append(name.replace('COMPANY', 'CO'))
            variations.append(name.replace('Company', 'Co'))

        if 'CORPORATION' in name.upper():
            variations.append(name.replace('CORPORATION', 'CORP'))
            variations.append(name.replace('Corporation', 'Corp'))

        if 'INCORPORATED' in name.upper():
            variations.append(name.replace('INCORPORATED', 'INC'))
            variations.append(name.replace('Incorporated', 'Inc'))

        if 'LIMITED' in name.upper():
            variations.append(name.replace('LIMITED', 'LTD'))
            variations.append(name.replace('Limited', 'Ltd'))

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var not in seen:
                seen.add(var)
                unique_variations.append(var)

        return unique_variations
