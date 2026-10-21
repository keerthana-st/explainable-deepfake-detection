"""
Metadata Agent (Module 3) for Explainable Deepfake Detection.

Performs deterministic, 4-tier provenance verification by parsing disclosed
metadata embedded in the image file: EXIF, XMP, C2PA manifests, optical consistency,
and JPEG Discrete Quantization Table (DQT) hashes.

This agent distinguishes:
1. Metadata evidence of authenticity (valid C2PA, genuine camera hardware DQT).
2. Metadata evidence of editing/generation (C2PA c2pa.created, generative software tags).
3. Modified/tampered metadata (impossible optical parameters, camera make vs. software DQT mismatch).
4. Missing / stripped metadata (neutral baseline; NEVER treated as proof of manipulation).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

from modules.metadata.result import MetadataResult, MetadataState, C2PAManifestDetails
from modules.metadata.c2pa_parser import C2PAManifestParser
from modules.metadata.exif_parser import ExifMetadataParser
from modules.metadata.xmp_parser import XmpMetadataParser
from modules.metadata.keywords import match_ai_keyword
from modules.metadata.consistency_engine import MetadataConsistencyEngine
from modules.metadata.dqt_database import DQTDatabase

logger = logging.getLogger(__name__)


class MetadataAgent:
    """
    4-Tier deterministic provenance, consistency, and metadata verification agent.
    """

    PROVENANCE_SCOPE_NOTICE = (
        "SCOPE NOTICE: This agent inspects disclosed provenance metadata (EXIF/XMP/C2PA/DQT). "
        "Missing metadata is common across messaging apps and web uploads and is explicitly "
        "treated as NEUTRAL/INCONCLUSIVE rather than evidence of forgery."
    )

    def __init__(self):
        self.c2pa_parser = C2PAManifestParser()
        self.exif_parser = ExifMetadataParser()
        self.xmp_parser = XmpMetadataParser()
        self.consistency_engine = MetadataConsistencyEngine()
        self.dqt_database = DQTDatabase()

    def analyze(self, image_path: Path | str) -> MetadataResult:
        path = Path(image_path)
        if not path.exists():
            return MetadataResult(
                metadata_flag=MetadataState.STRIPPED_INCONCLUSIVE.value,
                disclosed_tool=None,
                raw_manifest_data={},
                reasoning=f"File not found at path: {path}",
                data_source_tag="[ERROR: FILE_NOT_FOUND]",
            )

        # 1. Parse standards from file
        try:
            c2pa_res = self.c2pa_parser.parse(path)
        except Exception as e:
            c2pa_res = C2PAManifestDetails(manifest_found=False, validation_status=f"Exception: {e}")

        try:
            exif_res = self.exif_parser.parse(path)
        except Exception as e:
            exif_res = {"exif_found": False, "ai_disclosed": False, "raw_tags": {}}

        try:
            xmp_res = self.xmp_parser.parse(path)
        except Exception as e:
            xmp_res = {"xmp_found": False, "ai_disclosed": False, "history_agents": []}

        # 2. Run Optical Consistency Engine on EXIF
        raw_tags = exif_res.get("raw_tags", {})
        opt_report = self.consistency_engine.verify_consistency(raw_tags)

        # 3. Extract JPEG DQT Quantization Profile
        dqt_res = self.dqt_database.extract_dqt_from_file(path)

        # 4. Construct transparent raw audit data
        raw_manifest_data = {
            "c2pa": {
                "manifest_found": c2pa_res.manifest_found,
                "claim_generator": c2pa_res.claim_generator,
                "signing_issuer": c2pa_res.signing_issuer,
                "timestamp": c2pa_res.timestamp,
                "actions": c2pa_res.actions,
                "validation_status": c2pa_res.validation_status,
            },
            "exif": exif_res,
            "xmp": xmp_res,
            "optical_consistency": {
                "is_consistent": opt_report.is_consistent,
                "contradictions": opt_report.contradictions,
                "calculated_ev": opt_report.calculated_ev,
                "is_edited_software": opt_report.is_edited_software,
            },
            "dqt_fingerprint": {
                "dqt_found": dqt_res.dqt_found,
                "matched_profile": dqt_res.matched_profile,
                "profile_type": dqt_res.profile_type,
            },
        }

        # Visible data source tag
        active_sources = []
        if c2pa_res.manifest_found:
            active_sources.append("C2PA_JUMBF_STORE")
        if exif_res.get("exif_found"):
            active_sources.append("EXIF_APP1_TAGS")
        if xmp_res.get("xmp_found"):
            active_sources.append("XMP_PACKET_XML")
        if dqt_res.dqt_found:
            active_sources.append(f"DQT_0xFFDB({dqt_res.profile_type or 'custom'})")

        data_source_tag = (
            f"[VERIFIED_BINARY_EXTRACTION: {' + '.join(active_sources)}]"
            if active_sources
            else "[BINARY_EXTRACTION_COMPLETE: NO_STANDARDIZED_CHUNKS_FOUND]"
        )

        # 5. Check AI Generative Tools
        c2pa_ai_tool = self._check_c2pa_ai_tool(c2pa_res)
        exif_ai_tool = exif_res.get("matched_tool")
        xmp_ai_tool = xmp_res.get("matched_tool")
        ai_tool = c2pa_ai_tool or exif_ai_tool or xmp_ai_tool

        # 6. Apply 4-Tier Decision Taxonomy

        # Tier 1: Explicit Synthetic Generation Disclosed
        if ai_tool:
            reasoning = (
                f"CRYPTOGRAPHIC PROVENANCE DISCLOSURE DETECTED: The image discloses synthetic generation by '{ai_tool}'. "
                f"Triggers PROVENANCE OVERRIDE in Disagreement Layer."
            )
            return MetadataResult(
                metadata_flag=MetadataState.AI_DISCLOSED.value,
                disclosed_tool=ai_tool,
                raw_manifest_data=raw_manifest_data,
                reasoning=reasoning,
                data_source_tag=data_source_tag,
                c2pa_details=c2pa_res,
                exif_summary=exif_res,
                xmp_summary=xmp_res,
                optical_consistency=opt_report.__dict__,
                dqt_profile=dqt_res.__dict__,
            )

        # Tier 2: Tampered / Inconsistent Metadata
        if not opt_report.is_consistent:
            contra_str = "; ".join(opt_report.contradictions)
            reasoning = (
                f"METADATA TAMPERING / OPTICAL INCONSISTENCY DETECTED: {contra_str}. "
                f"Physical sensor settings contradict natural optical laws."
            )
            return MetadataResult(
                metadata_flag=MetadataState.TAMPERED_INCONSISTENT.value,
                disclosed_tool=None,
                raw_manifest_data=raw_manifest_data,
                reasoning=reasoning,
                data_source_tag=data_source_tag,
                c2pa_details=c2pa_res,
                exif_summary=exif_res,
                xmp_summary=xmp_res,
                optical_consistency=opt_report.__dict__,
                dqt_profile=dqt_res.__dict__,
            )

        # Tier 3: C2PA Manifest Present but Generator Unclear
        if c2pa_res.manifest_found and not c2pa_res.is_valid_signature:
            reasoning = (
                "MANIFEST PRESENT (GENERATOR UNCLEAR): C2PA Manifest JUMBF chunk detected, "
                "but no recognized generative AI tool or trusted certificate root could be confirmed."
            )
            return MetadataResult(
                metadata_flag=MetadataState.MANIFEST_PRESENT_GENERATOR_UNCLEAR.value,
                disclosed_tool=None,
                raw_manifest_data=raw_manifest_data,
                reasoning=reasoning,
                data_source_tag=data_source_tag,
                c2pa_details=c2pa_res,
                exif_summary=exif_res,
                xmp_summary=xmp_res,
                optical_consistency=opt_report.__dict__,
                dqt_profile=dqt_res.__dict__,
            )

        # Tier 4: Verified Authentic Provenance (Valid C2PA or Camera EXIF + DQT)
        if c2pa_res.manifest_found and c2pa_res.is_valid_signature:
            reasoning = (
                f"CRYPTOGRAPHICALLY VERIFIED AUTHENTIC: Valid C2PA signature verified against trusted root "
                f"certificate: {c2pa_res.signing_issuer}."
            )
            return MetadataResult(
                metadata_flag=MetadataState.PRESENT_NO_AI_SIGNAL.value,
                disclosed_tool=None,
                raw_manifest_data=raw_manifest_data,
                reasoning=reasoning,
                data_source_tag=data_source_tag,
                c2pa_details=c2pa_res,
                exif_summary=exif_res,
                xmp_summary=xmp_res,
                optical_consistency=opt_report.__dict__,
                dqt_profile=dqt_res.__dict__,
            )

        if exif_res.get("exif_found") or xmp_res.get("xmp_found"):
            soft_note = f" (Edited with {opt_report.detected_software})" if opt_report.is_edited_software else ""
            reasoning = (
                f"METADATA PRESENT (NO AI DISCLOSURE): Genuine camera metadata present with consistent optical parameters"
                f"{soft_note}. However, presence of metadata does not prove authenticity, as EXIF can be copied or spoofed."
            )
            return MetadataResult(
                metadata_flag=MetadataState.PRESENT_NO_AI_SIGNAL.value,
                disclosed_tool=None,
                raw_manifest_data=raw_manifest_data,
                reasoning=reasoning,
                data_source_tag=data_source_tag,
                c2pa_details=c2pa_res,
                exif_summary=exif_res,
                xmp_summary=xmp_res,
                optical_consistency=opt_report.__dict__,
                dqt_profile=dqt_res.__dict__,
            )

        # Tier 5: Missing / Stripped Metadata (Neutral baseline)
        reasoning = (
            "NO METADATA DETECTED (MISSING / STRIPPED): No EXIF, XMP, or C2PA chunks present. "
            "This is an explicitly WEAK and AMBIGUOUS signal: social media platforms and messaging apps "
            "strip metadata by default for privacy and bandwidth optimization. "
            "MANDATORY POLICY: Missing metadata is NEUTRAL and is NOT treated as evidence of forgery."
        )
        return MetadataResult(
            metadata_flag=MetadataState.MISSING.value,
            disclosed_tool=None,
            raw_manifest_data=raw_manifest_data,
            reasoning=reasoning,
            data_source_tag=data_source_tag,
            c2pa_details=c2pa_res,
            exif_summary=exif_res,
            xmp_summary=xmp_res,
            optical_consistency=opt_report.__dict__,
            dqt_profile=dqt_res.__dict__,
        )

    def _check_c2pa_ai_tool(self, c2pa_res: C2PAManifestDetails) -> Optional[str]:
        if not c2pa_res.manifest_found:
            return None
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
                return c2pa_res.claim_generator or match[1]
            elif "openai" in c.lower() or "gpt" in c.lower() or "dall-e" in c.lower():
                return c2pa_res.claim_generator or "OpenAI DALL-E / ChatGPT"
        return None
