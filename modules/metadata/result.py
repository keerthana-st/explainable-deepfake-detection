"""
Data models and enumeration representing Metadata Agent extraction results.

Supports 4-tier provenance taxonomy:
1. VERIFIED_AUTHENTIC_PROVENANCE: Cryptographic hardware root or verified camera metadata.
2. EXPLICIT_SYNTHETIC_PROVENANCE: Known generative AI software tags or C2PA synthetic origin.
3. METADATA_TAMPERED_INCONSISTENT: Internal optical contradictions or DQT hardware mismatches.
4. METADATA_STRIPPED_INCONCLUSIVE: Zero EXIF or stripped web headers. (NEVER treated as fake).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MetadataState(str, Enum):
    """
    Enhanced four-tier provenance classification states.
    """
    AI_DISCLOSED = "ai_disclosed"
    MANIFEST_PRESENT_GENERATOR_UNCLEAR = "manifest_present_generator_unclear"
    PRESENT_NO_AI_SIGNAL = "present_no_ai_signal"
    MISSING = "missing"
    TAMPERED_INCONSISTENT = "tampered_inconsistent"

    # Enhanced aliases
    EXPLICIT_SYNTHETIC = "ai_disclosed"
    VERIFIED_AUTHENTIC = "present_no_ai_signal"
    STRIPPED_INCONCLUSIVE = "missing"


@dataclass
class C2PAManifestDetails:
    """Structured details extracted from a C2PA Content Credentials manifest."""
    manifest_found: bool = False
    claim_generator: Optional[str] = None
    title: Optional[str] = None
    format: Optional[str] = None
    instance_id: Optional[str] = None
    actions: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: Optional[str] = None
    signing_issuer: Optional[str] = None
    is_valid_signature: Optional[bool] = None
    validation_status: Optional[str] = None


@dataclass
class MetadataResult:
    """Complete explainable result from MetadataAgent analysis."""

    # 1. Primary Classification Flag (one of MetadataState values)
    metadata_flag: str

    # 2. Specific Disclosed Tool (e.g. 'Midjourney', 'DALL-E 3', or None)
    disclosed_tool: Optional[str]

    # 3. Transparent Raw Parsed Data for forensic audit
    raw_manifest_data: Dict[str, Any]

    # 4. State-calibrated explainable reasoning
    reasoning: str

    # 5. Visible cryptographic/file parsing audit tag
    data_source_tag: str

    # 6. Structured C2PA details
    c2pa_details: C2PAManifestDetails = field(default_factory=C2PAManifestDetails)

    # 7. Parsed EXIF summary
    exif_summary: Dict[str, Any] = field(default_factory=dict)

    # 8. Parsed XMP summary
    xmp_summary: Dict[str, Any] = field(default_factory=dict)

    # 9. Optical Consistency Report
    optical_consistency: Dict[str, Any] = field(default_factory=dict)

    # 10. JPEG DQT Table Matching Profile
    dqt_profile: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_unclear(self) -> bool:
        return self.metadata_flag == MetadataState.MANIFEST_PRESENT_GENERATOR_UNCLEAR.value

    @property
    def is_ai_disclosed(self) -> bool:
        return self.metadata_flag in [
            MetadataState.AI_DISCLOSED.value,
            MetadataState.EXPLICIT_SYNTHETIC.value,
        ]

    @property
    def is_missing(self) -> bool:
        return self.metadata_flag in [
            MetadataState.MISSING.value,
            MetadataState.STRIPPED_INCONCLUSIVE.value,
        ]

    @property
    def is_tampered(self) -> bool:
        return self.metadata_flag == MetadataState.TAMPERED_INCONSISTENT.value

    @property
    def is_authentic_supporting(self) -> bool:
        return self.metadata_flag in [
            MetadataState.PRESENT_NO_AI_SIGNAL.value,
            MetadataState.VERIFIED_AUTHENTIC.value,
        ]
