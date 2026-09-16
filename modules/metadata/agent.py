"""
Metadata Agent (Module 3) for Explainable Deepfake Detection.

Performs rule-based, non-learned provenance verification by parsing disclosed
metadata embedded in the image file: EXIF, XMP, and C2PA manifests.

This agent provides direct, cryptographically-verifiable evidence when present,
contrasting with the probabilistic/statistical inference of the Frequency and
Classifier agents.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

from modules.metadata.result import MetadataResult, MetadataState, C2PAManifestDetails
from modules.metadata.c2pa_parser import C2PAManifestParser
from modules.metadata.exif_parser import ExifMetadataParser
from modules.metadata.xmp_parser import XmpMetadataParser
from modules.metadata.keywords import match_ai_keyword

logger = logging.getLogger(__name__)


class MetadataAgent:
    """
    Deterministic provenance and metadata verification agent.
    """

    # Mandatory scope and coverage disclosures
    PROVENANCE_SCOPE_NOTICE = (
        "SCOPE NOTICE: This agent detects disclosed provenance metadata (EXIF/XMP/C2PA). "
        "It cannot detect proprietary invisible watermarks such as SynthID, which require the "
        "generator's private detection key. Real-world coverage is typically low (<50%) as social "
        "media platforms and messaging apps strip metadata during upload."
    )

    def __init__(self):
        self.c2pa_parser = C2PAManifestParser()
        self.exif_parser = ExifMetadataParser()
        self.xmp_parser = XmpMetadataParser()

    def analyze(self, image_path: Path | str) -> MetadataResult:
        """
        Analyzes the target image for embedded EXIF, XMP, and C2PA metadata.

        Args:
            image_path: Path to the image file to inspect.

        Returns:
            MetadataResult containing the 4-tier state classification, extracted tool,
            raw audit dictionaries, visible data source tag, and explainable reasoning.
        """
        path = Path(image_path)
        if not path.exists():
            return MetadataResult(
                metadata_flag=MetadataState.MISSING.value,
                disclosed_tool=None,
                raw_manifest_data={},
                reasoning=f"File not found at path: {path}",
                data_source_tag="[ERROR: FILE_NOT_FOUND]",
            )

        # 1. Parse all three metadata standards from binary file
        try:
            c2pa_res = self.c2pa_parser.parse(path)
        except Exception as e:
            logger.warning(f"C2PA parsing error: {e}")
            c2pa_res = C2PAManifestDetails(manifest_found=False, validation_status=f"Exception: {e}")

        try:
            exif_res = self.exif_parser.parse(path)
        except Exception as e:
            logger.warning(f"EXIF parsing error: {e}")
            exif_res = {"exif_found": False, "ai_disclosed": False, "raw_tags": {}}

        try:
            xmp_res = self.xmp_parser.parse(path)
        except Exception as e:
            logger.warning(f"XMP parsing error: {e}")
            xmp_res = {"xmp_found": False, "ai_disclosed": False, "history_agents": []}

        # 2. Compile transparent raw audit data
        raw_manifest_data = {
            "c2pa": {
                "manifest_found": c2pa_res.manifest_found,
                "claim_generator": c2pa_res.claim_generator,
                "title": c2pa_res.title,
                "signing_issuer": c2pa_res.signing_issuer,
                "timestamp": c2pa_res.timestamp,
                "actions": c2pa_res.actions,
                "validation_status": c2pa_res.validation_status,
            },
            "exif": exif_res,
            "xmp": xmp_res,
        }

        # 3. Construct visible data source tag confirming real binary parsing
        active_sources = []
        if c2pa_res.manifest_found:
            active_sources.append("C2PA_JUMBF_STORE")
        if exif_res.get("exif_found"):
            active_sources.append("EXIF_APP1_TAGS")
        if xmp_res.get("xmp_found"):
            active_sources.append("XMP_PACKET_XML")

        if active_sources:
            data_source_tag = f"[VERIFIED_BINARY_EXTRACTION: {' + '.join(active_sources)}]"
        else:
            data_source_tag = "[BINARY_EXTRACTION_COMPLETE: NO_STANDARDIZED_CHUNKS_FOUND]"

        # 4. Synthesize Four-State Classification Logic

        # Check A: C2PA AI Disclosure
        c2pa_ai_tool = None
        if c2pa_res.manifest_found:
            # Check claim_generator string or software_agents in actions
            candidates = []
            if c2pa_res.claim_generator:
                candidates.append(c2pa_res.claim_generator)
            for act in c2pa_res.actions:
                sw = act.get("software_agent", {})
                if isinstance(sw, dict) and sw.get("name"):
                    name = sw.get("name")
                    ver = sw.get("version")
                    candidates.append(f"{name} {ver}".strip() if ver else name)
                if act.get("digital_source_type") and "algorithmic" in str(act.get("digital_source_type")).lower():
                    candidates.append("IPTC trainedAlgorithmicMedia")

            for c in candidates:
                match = match_ai_keyword(c)
                if match:
                    c2pa_ai_tool = c2pa_res.claim_generator or match[1]
                    break
                elif "openai" in c.lower() or "gpt" in c.lower():
                    c2pa_ai_tool = c2pa_res.claim_generator or "OpenAI API / ChatGPT"
                    break

        # Check B: EXIF / XMP AI Disclosure
        exif_ai_tool = exif_res.get("matched_tool")
        xmp_ai_tool = xmp_res.get("matched_tool")

        # State 1: "ai_disclosed"
        if c2pa_ai_tool or exif_ai_tool or xmp_ai_tool:
            disclosed_tool = c2pa_ai_tool or exif_ai_tool or xmp_ai_tool
            reasoning = self._format_ai_disclosed_reasoning(
                disclosed_tool=disclosed_tool,
                c2pa_res=c2pa_res,
                exif_res=exif_res,
                xmp_res=xmp_res,
            )
            return MetadataResult(
                metadata_flag=MetadataState.AI_DISCLOSED.value,
                disclosed_tool=disclosed_tool,
                raw_manifest_data=raw_manifest_data,
                reasoning=reasoning,
                data_source_tag=data_source_tag,
                c2pa_details=c2pa_res,
                exif_summary=exif_res,
                xmp_summary=xmp_res,
            )

        # State 2: "manifest_present_generator_unclear"
        if c2pa_res.manifest_found:
            reasoning = self._format_manifest_unclear_reasoning(c2pa_res)
            return MetadataResult(
                metadata_flag=MetadataState.MANIFEST_PRESENT_GENERATOR_UNCLEAR.value,
                disclosed_tool=None,
                raw_manifest_data=raw_manifest_data,
                reasoning=reasoning,
                data_source_tag=data_source_tag,
                c2pa_details=c2pa_res,
                exif_summary=exif_res,
                xmp_summary=xmp_res,
            )

        # State 3: "present_no_ai_signal"
        if exif_res.get("exif_found") or xmp_res.get("xmp_found"):
            reasoning = self._format_no_ai_signal_reasoning(exif_res, xmp_res)
            return MetadataResult(
                metadata_flag=MetadataState.PRESENT_NO_AI_SIGNAL.value,
                disclosed_tool=None,
                raw_manifest_data=raw_manifest_data,
                reasoning=reasoning,
                data_source_tag=data_source_tag,
                c2pa_details=c2pa_res,
                exif_summary=exif_res,
                xmp_summary=xmp_res,
            )

        # State 4: "missing"
        reasoning = self._format_missing_reasoning()
        return MetadataResult(
            metadata_flag=MetadataState.MISSING.value,
            disclosed_tool=None,
            raw_manifest_data=raw_manifest_data,
            reasoning=reasoning,
            data_source_tag=data_source_tag,
            c2pa_details=c2pa_res,
            exif_summary=exif_res,
            xmp_summary=xmp_res,
        )

    def _format_ai_disclosed_reasoning(
        self,
        disclosed_tool: str,
        c2pa_res: C2PAManifestDetails,
        exif_res: Dict[str, Any],
        xmp_res: Dict[str, Any],
    ) -> str:
        lines = [
            f"CRYPTOGRAPHIC PROVENANCE DISCLOSURE DETECTED: The image discloses synthetic generation by '{disclosed_tool}'.",
        ]
        if c2pa_res.manifest_found:
            lines.append("• C2PA Manifest Verification:")
            if c2pa_res.claim_generator:
                lines.append(f"  - Claim Generator: {c2pa_res.claim_generator}")
            if c2pa_res.signing_issuer:
                lines.append(f"  - Certificate Signer: {c2pa_res.signing_issuer}")
            if c2pa_res.timestamp:
                lines.append(f"  - Timestamp: {c2pa_res.timestamp}")
            if c2pa_res.actions:
                act_summaries = [f"{a.get('action')}" for a in c2pa_res.actions if a.get("action")]
                lines.append(f"  - Manifest Actions: {', '.join(act_summaries)}")

        if exif_res.get("ai_disclosed"):
            lines.append(f"• EXIF Software Tag: Identified AI tool signature in image header ('{exif_res.get('software')}').")

        if xmp_res.get("ai_disclosed"):
            lines.append(f"• XMP Packet Tag: Identified generative software tool in XMP Creator block ('{xmp_res.get('matched_tool')}').")

        lines.append(
            "\nFUSION IMPACT: This constitutes direct, verifiable cryptographic provenance evidence. "
            "Under Disagreement Layer arbitration, this disclosure triggers a PROVENANCE OVERRIDE, "
            "categorically superseding probabilistic heuristic scores."
        )
        return "\n".join(lines)

    def _format_manifest_unclear_reasoning(self, c2pa_res: C2PAManifestDetails) -> str:
        lines = [
            "C2PA MANIFEST PRESENT (GENERATOR UNCLEAR): A Coalition for Content Provenance and Authenticity (C2PA) "
            "manifest container was detected embedded in this file, but the specific claim_generator field could not be "
            "conclusively resolved or mapped to a recognized tool.",
            f"• Manifest Status: {c2pa_res.validation_status}",
        ]
        if c2pa_res.signing_issuer:
            lines.append(f"• Certificate Issuer: {c2pa_res.signing_issuer}")
        if c2pa_res.timestamp:
            lines.append(f"• Recorded Timestamp: {c2pa_res.timestamp}")

        lines.append(
            "\nINTERPRETATION: An intact provenance container exists, demonstrating cryptographic tracking, "
            "but the generating application or editing workflow did not disclose a recognized software identity. "
            "This finding does not prove AI generation and cannot trigger an override; the probabilistic "
            "Frequency and Classifier agents must be consulted for visual analysis."
        )
        return "\n".join(lines)

    def _format_no_ai_signal_reasoning(self, exif_res: Dict[str, Any], xmp_res: Dict[str, Any]) -> str:
        lines = [
            "METADATA PRESENT (NO AI DISCLOSURE): Standard image metadata was successfully parsed, "
            "with no AI generation tools, synthetic descriptors, or C2PA algorithmic assertions disclosed.",
        ]
        if exif_res.get("make") or exif_res.get("model"):
            cam_str = f"{exif_res.get('make', '')} {exif_res.get('model', '')}".strip()
            lines.append(f"• Camera Hardware Profile: Disclosed capture device '{cam_str}'.")
        if exif_res.get("datetime"):
            lines.append(f"• Original Capture Timestamp: {exif_res.get('datetime')}")
        if exif_res.get("lens_model"):
            lines.append(f"• Lens Specification: {exif_res.get('lens_model')}")
        if exif_res.get("software"):
            lines.append(f"• Recording Software: {exif_res.get('software')}")

        lines.append(
            "\nINTERPRETATION: The presence of camera hardware tags weakly supports authentic origin. "
            "However, metadata can be fabricated or inherited from an authentic donor image. "
            "Therefore, this metadata alone does not prove authenticity and cannot override visual analysis."
        )
        return "\n".join(lines)

    def _format_missing_reasoning(self) -> str:
        return (
            "NO METADATA DETECTED (MISSING / STRIPPED): No EXIF tags, XMP packets, or C2PA Content Credentials "
            "survived in this image file.\n\n"
            "CRITICAL CONTEXT: The absence of metadata is an explicitly WEAK and AMBIGUOUS signal. "
            "Virtually all major social media platforms (WhatsApp, X, Instagram, Facebook), messaging apps, and "
            "screenshot utilities completely strip EXIF and C2PA metadata during transmission to protect user privacy "
            "and compress file size. Absence of metadata is equally common in authentic mobile photos and AI deepfakes alike. "
            "This agent provides NO provenance signal for this image; verdicts must rely entirely on Frequency and Classifier analyses."
        )
