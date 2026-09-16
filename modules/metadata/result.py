"""
Data models and enumeration representing Metadata Agent extraction results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class MetadataState(str, Enum):
    """
    Four distinct result classification states for image provenance metadata.
    """
    AI_DISCLOSED = "ai_disclosed"
    MANIFEST_PRESENT_GENERATOR_UNCLEAR = "manifest_present_generator_unclear"
    PRESENT_NO_AI_SIGNAL = "present_no_ai_signal"
    MISSING = "missing"


@dataclass
class C2PAManifestDetails:
    """
    Structured details extracted from a C2PA Content Credentials manifest.
    """
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
    """
    Complete explainable result from MetadataAgent analysis.
    """
    # 1. Primary Classification Flag
    metadata_flag: str  # One of MetadataState values

    # 2. Specific Disclosed Tool (e.g. 'ChatGPT c2pa-rs/0.28.4', 'Midjourney', or None)
    disclosed_tool: Optional[str]

    # 3. Transparent Raw Parsed Data (EXIF, XMP, C2PA) for forensic audit
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

    @property
    def is_ai_disclosed(self) -> bool:
        return self.metadata_flag == MetadataState.AI_DISCLOSED.value

    @property
    def is_missing(self) -> bool:
        return self.metadata_flag == MetadataState.MISSING.value

    @property
    def is_unclear(self) -> bool:
        return self.metadata_flag == MetadataState.MANIFEST_PRESENT_GENERATOR_UNCLEAR.value

    @property
    def is_authentic_supporting(self) -> bool:
        return self.metadata_flag == MetadataState.PRESENT_NO_AI_SIGNAL.value
