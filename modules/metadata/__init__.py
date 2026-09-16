"""
Metadata Agent Module (Module 3) for Explainable Deepfake Detection.
"""

from modules.metadata.agent import MetadataAgent
from modules.metadata.result import MetadataResult, MetadataState, C2PAManifestDetails
from modules.metadata.keywords import AI_TOOL_KEYWORDS, match_ai_keyword

__all__ = [
    "MetadataAgent",
    "MetadataResult",
    "MetadataState",
    "C2PAManifestDetails",
    "AI_TOOL_KEYWORDS",
    "match_ai_keyword",
]
