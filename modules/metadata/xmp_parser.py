"""
XMP (Extensible Metadata Platform) Packet Parser.

Extracts Adobe and industry-standard XMP data packets embedded in image files
and inspects CreatorTool, history software agents, and description tags.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import re
import xml.etree.ElementTree as ET

from modules.metadata.keywords import match_ai_keyword

logger = logging.getLogger(__name__)


class XmpMetadataParser:
    """
    Scans and extracts XMP metadata blocks from binary streams and XML payloads.
    """

    def parse(self, image_path: Path) -> Dict[str, Any]:
        """
        Extracts XMP metadata from image file.

        Args:
            image_path: Path to image file.

        Returns:
            Dictionary containing:
                - xmp_found: bool
                - creator_tool: Optional[str]
                - creator: Optional[str]
                - history_agents: List[str]
                - ai_disclosed: bool
                - matched_tool: Optional[str]
                - raw_xmp: Optional[str]
        """
        image_path = Path(image_path)
        result = {
            "xmp_found": False,
            "creator_tool": None,
            "creator": None,
            "history_agents": [],
            "ai_disclosed": False,
            "matched_tool": None,
            "raw_xmp": None,
        }

        if not image_path.exists():
            return result

        raw_xmp_text = self._extract_xmp_packet(image_path)
        if not raw_xmp_text:
            return result

        result["xmp_found"] = True
        result["raw_xmp"] = raw_xmp_text[:2000]  # Store snippet for audit

        # Parse XMP XML tags safely
        try:
            # Clean namespace prefixes for easier parsing if needed
            # Search for CreatorTool
            creator_match = re.search(r"<[^>]*CreatorTool[^>]*>([^<]+)<\/[^>]*CreatorTool>", raw_xmp_text, re.IGNORECASE)
            if not creator_match:
                creator_match = re.search(r'xmp:CreatorTool="([^"]+)"', raw_xmp_text, re.IGNORECASE)
            if creator_match:
                result["creator_tool"] = creator_match.group(1).strip()

            # Search for dc:creator
            author_match = re.search(r"<dc:creator>[\s\S]*?<rdf:li>([^<]+)<\/rdf:li>[\s\S]*?<\/dc:creator>", raw_xmp_text, re.IGNORECASE)
            if author_match:
                result["creator"] = author_match.group(1).strip()

            # Search for software agents in History
            history_matches = re.findall(r'stEvt:softwareAgent="([^"]+)"', raw_xmp_text, re.IGNORECASE)
            result["history_agents"] = [h.strip() for h in history_matches if h.strip()]

        except Exception as e:
            logger.debug(f"XMP XML parsing error: {e}")

        # Check for AI tool disclosures
        targets = [
            result["creator_tool"],
            result["creator"],
        ] + result["history_agents"]

        for target in targets:
            if target:
                match = match_ai_keyword(target)
                if match:
                    result["ai_disclosed"] = True
                    result["matched_tool"] = match[1]
                    break

        # Also regex scan the whole packet for known keywords if not yet matched
        if not result["ai_disclosed"]:
            match = match_ai_keyword(raw_xmp_text)
            if match:
                result["ai_disclosed"] = True
                result["matched_tool"] = match[1]

        return result

    def _extract_xmp_packet(self, image_path: Path) -> Optional[str]:
        """
        Locates and extracts XMP XML packet from image file bytes.
        """
        try:
            with open(image_path, "rb") as f:
                content = f.read()

            # 1. Search for standard XMP packet markers <?xpacket ... ?>
            start_idx = content.find(b"<?xpacket begin")
            if start_idx == -1:
                start_idx = content.find(b"<x:xmpmeta")

            if start_idx != -1:
                end_idx = content.find(b"<?xpacket end", start_idx)
                if end_idx != -1:
                    packet_bytes = content[start_idx:end_idx + 19]
                else:
                    end_close = content.find(b"</x:xmpmeta>", start_idx)
                    if end_close != -1:
                        packet_bytes = content[start_idx:end_close + 12]
                    else:
                        packet_bytes = content[start_idx:start_idx + 32768]
                return packet_bytes.decode("utf-8", errors="ignore")

        except Exception as e:
            logger.debug(f"Error extracting XMP packet: {e}")

        return None
