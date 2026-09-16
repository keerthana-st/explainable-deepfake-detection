"""
C2PA (Coalition for Content Provenance and Authenticity) Manifest Parser.

Extracts cryptographic content credentials, claim generators, actions,
timestamps, and certificate signing authorities from image files.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import logging
import re
import struct

from modules.metadata.result import C2PAManifestDetails
from modules.metadata.keywords import match_ai_keyword

logger = logging.getLogger(__name__)

# Try importing the official c2pa-python library
try:
    import c2pa
    HAS_C2PA_LIB = True
except ImportError:
    HAS_C2PA_LIB = False
    c2pa = None


class C2PAManifestParser:
    """
    Parser for C2PA provenance manifests embedded in JPEG, PNG, WebP, and TIFF files.
    """

    def __init__(self):
        self.has_native_lib = HAS_C2PA_LIB

    def parse(self, image_path: Path) -> C2PAManifestDetails:
        """
        Parses C2PA manifest from the target image file.

        Args:
            image_path: Absolute or relative Path to image file.

        Returns:
            C2PAManifestDetails populated with parsed fields.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return C2PAManifestDetails(manifest_found=False, validation_status="File not found")

        # 1. Primary path: Official C2PA SDK
        if self.has_native_lib:
            try:
                return self._parse_with_sdk(image_path)
            except Exception as e:
                err_str = str(e)
                # Check if it was purely "no manifest found"
                if "ManifestNotFound" in err_str or "no JUMBF data found" in err_str:
                    # Double check if any raw C2PA chunk exists that was somehow missed
                    has_raw_box, raw_type = self._check_raw_c2pa_chunks(image_path)
                    if has_raw_box:
                        return C2PAManifestDetails(
                            manifest_found=True,
                            validation_status=f"Raw C2PA box ({raw_type}) detected but manifest unparseable by SDK: {err_str}",
                        )
                    return C2PAManifestDetails(
                        manifest_found=False,
                        validation_status="No C2PA manifest found",
                    )
                else:
                    # SDK error on malformed/corrupted file
                    logger.warning(f"C2PA SDK parsing warning on {image_path.name}: {e}")
                    has_raw_box, raw_type = self._check_raw_c2pa_chunks(image_path)
                    if has_raw_box:
                        # Fallback to direct raw box scanner
                        return self._fallback_raw_box_parse(image_path, raw_type)
                    return C2PAManifestDetails(
                        manifest_found=False,
                        validation_status=f"C2PA SDK error: {err_str}",
                    )

        # 2. Secondary path: Fallback pure-Python box parser if SDK is unavailable
        has_raw_box, raw_type = self._check_raw_c2pa_chunks(image_path)
        if has_raw_box:
            return self._fallback_raw_box_parse(image_path, raw_type)

        return C2PAManifestDetails(
            manifest_found=False,
            validation_status="C2PA SDK unavailable and no raw C2PA chunks found",
        )

    def _parse_with_sdk(self, image_path: Path) -> C2PAManifestDetails:
        """Uses c2pa.Reader to extract detailed crJSON manifest."""
        details = C2PAManifestDetails()

        with c2pa.Reader(str(image_path)) as reader:
            manifest_json_str = reader.json()
            data = json.loads(manifest_json_str)

        active_manifest_id = data.get("active_manifest")
        manifests = data.get("manifests", {})
        active = manifests.get(active_manifest_id, {}) if active_manifest_id else {}

        if not active and manifests:
            # Pick first manifest if active_manifest is not set
            active = next(iter(manifests.values()))

        if not active:
            details.manifest_found = False
            details.validation_status = "Manifest container empty"
            return details

        details.manifest_found = True
        details.title = active.get("title")
        details.format = active.get("format")
        details.instance_id = active.get("instance_id")

        # 1. Extract claim_generator string
        # C2PA manifests may store this in 'claim_generator' (v1) or 'claim_generator_info' (v2)
        generator_str = active.get("claim_generator")
        gen_info_list = active.get("claim_generator_info", [])

        extracted_generators = []
        if generator_str:
            extracted_generators.append(str(generator_str))

        if isinstance(gen_info_list, list):
            for g in gen_info_list:
                if isinstance(g, dict):
                    name = g.get("name")
                    version = g.get("version") or g.get("specVersion")
                    c2pa_rs = g.get("org.contentauth.c2pa_rs")
                    parts = [name] if name else []
                    if version:
                        parts.append(f"v{version}")
                    if c2pa_rs:
                        parts.append(f"(c2pa-rs/{c2pa_rs})")
                    if parts:
                        extracted_generators.append(" ".join(parts))

        # 2. Extract Actions and SoftwareAgent from assertions
        actions_list = []
        software_agents = []
        digital_source_types = []

        assertions = active.get("assertions", [])
        for assertion in assertions:
            label = assertion.get("label", "")
            data_body = assertion.get("data", {})

            if "action" in label and isinstance(data_body, dict):
                raw_actions = data_body.get("actions", [])
                for act in raw_actions:
                    if isinstance(act, dict):
                        act_type = act.get("action", "")
                        when = act.get("when")
                        sw_agent = act.get("softwareAgent", {})
                        ds_type = act.get("digitalSourceType")

                        if isinstance(sw_agent, dict):
                            sw_name = sw_agent.get("name")
                            sw_ver = sw_agent.get("version")
                            if sw_name:
                                full_sw = f"{sw_name} {sw_ver}".strip() if sw_ver else sw_name
                                software_agents.append(full_sw)

                        if ds_type:
                            digital_source_types.append(ds_type)

                        actions_list.append({
                            "action": act_type,
                            "timestamp": when,
                            "software_agent": sw_agent,
                            "digital_source_type": ds_type,
                        })

        details.actions = actions_list

        # Combine generator candidates (never using 'C2PA Content Credentials' placeholder)
        all_gen_candidates = extracted_generators + software_agents
        # Filter out generic placeholder phrases if present
        filtered_candidates = [
            c for c in all_gen_candidates
            if c.strip() and "c2pa content credentials" not in c.lower()
        ]

        if filtered_candidates:
            details.claim_generator = " / ".join(dict.fromkeys(filtered_candidates))
        elif all_gen_candidates:
            details.claim_generator = all_gen_candidates[0]

        # 3. Extract Signature Info (Cert Issuer, Signing Time, Algorithm)
        sig_info = active.get("signature_info", {})
        if isinstance(sig_info, dict):
            details.signing_issuer = sig_info.get("issuer") or sig_info.get("common_name")
            details.timestamp = sig_info.get("time")
            details.is_valid_signature = True  # c2pa.Reader validates signature on load
            details.validation_status = "Valid C2PA Signature"
        else:
            details.validation_status = "Manifest present without standard signature block"

        return details

    def _check_raw_c2pa_chunks(self, image_path: Path) -> Tuple[bool, str]:
        """
        Quick byte scanner to check for PNG 'caBX' chunk or JPEG APP11 / JUMBF box.
        """
        try:
            with open(image_path, "rb") as f:
                header = f.read(16)
                if not header:
                    return False, "empty"

                # Check PNG
                if header.startswith(b"\x89PNG\r\n\x1a\n"):
                    f.seek(8)
                    while True:
                        length_bytes = f.read(4)
                        if len(length_bytes) < 4:
                            break
                        length = struct.unpack(">I", length_bytes)[0]
                        chunk_type = f.read(4)
                        if chunk_type == b"caBX":
                            return True, "PNG_caBX"
                        if chunk_type == b"IEND" or len(chunk_type) < 4:
                            break
                        f.seek(length + 4, 1)

                # Check JPEG APP11 (0xFF, 0xEB) or JUMBF marker 'jumd'
                elif header.startswith(b"\xFF\xD8"):
                    f.seek(2)
                    while True:
                        marker_bytes = f.read(2)
                        if len(marker_bytes) < 2 or marker_bytes[0] != 0xFF:
                            break
                        marker = marker_bytes[1]
                        if marker in (0xD9, 0xDA):  # EOI or SOS
                            break
                        len_bytes = f.read(2)
                        if len(len_bytes) < 2:
                            break
                        seg_len = struct.unpack(">H", len_bytes)[0]
                        if marker == 0xEB:  # APP11 (C2PA/JUMBF)
                            payload = f.read(min(seg_len - 2, 64))
                            if b"JP2" in payload or b"jumd" in payload or b"c2pa" in payload:
                                return True, "JPEG_APP11_JUMBF"
                            f.seek(seg_len - 2 - len(payload), 1)
                        else:
                            f.seek(seg_len - 2, 1)

                # Check WebP / RIFF
                elif header.startswith(b"RIFF") and b"WEBP" in header:
                    f.seek(12)
                    while True:
                        chunk_id = f.read(4)
                        if len(chunk_id) < 4:
                            break
                        chunk_sz_bytes = f.read(4)
                        if len(chunk_sz_bytes) < 4:
                            break
                        chunk_sz = struct.unpack("<I", chunk_sz_bytes)[0]
                        if chunk_id == b"c2pa":
                            return True, "WEBP_c2pa"
                        f.seek(chunk_sz + (chunk_sz % 2), 1)

        except Exception as e:
            logger.debug(f"Error during raw chunk check: {e}")

        return False, "none"

    def _fallback_raw_box_parse(self, image_path: Path, box_type: str) -> C2PAManifestDetails:
        """
        Pure-Python fallback extractor that scans for JUMBF strings when SDK fails or is unavailable.
        """
        details = C2PAManifestDetails(manifest_found=True, validation_status=f"Parsed via Fallback Scanner ({box_type})")
        try:
            with open(image_path, "rb") as f:
                content = f.read()

            # Find all printable alphanumeric substrings
            strings = re.findall(b"[a-zA-Z0-9_\\-\\.\\/ :]{4,}", content)
            candidates = []
            issuer = None

            for b_str in strings:
                try:
                    s = b_str.decode("utf-8", errors="ignore").strip()
                    s_lower = s.lower()
                    if any(k in s_lower for k in ["chatgpt", "dall-e", "openai", "midjourney", "firefly", "gpt-image", "flux"]):
                        if s not in candidates and "content credentials" not in s_lower:
                            candidates.append(s)
                    if "opco" in s_lower or "openai media" in s_lower:
                        issuer = s
                except Exception:
                    pass

            if candidates:
                details.claim_generator = candidates[0]
            if issuer:
                details.signing_issuer = issuer

        except Exception as e:
            details.validation_status = f"Fallback parse error: {e}"

        return details
