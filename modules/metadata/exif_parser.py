"""
EXIF (Exchangeable Image File Format) Metadata Parser.

Extracts standard camera metadata, hardware provenance, and software tags
from images using Pillow and ExifRead.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from PIL import Image, ExifTags
import exifread

from modules.metadata.keywords import match_ai_keyword

logger = logging.getLogger(__name__)


class ExifMetadataParser:
    """
    Parses EXIF tags and checks for known AI generation and editing software signatures.
    """

    def parse(self, image_path: Path) -> Dict[str, Any]:
        """
        Extracts EXIF metadata from image file.

        Args:
            image_path: Path to image file.

        Returns:
            Dictionary containing:
                - exif_found: bool
                - software: Optional[str]
                - make: Optional[str]
                - model: Optional[str]
                - datetime: Optional[str]
                - ai_disclosed: bool
                - matched_tool: Optional[str]
                - raw_tags: Dict[str, str]
        """
        image_path = Path(image_path)
        result = {
            "exif_found": False,
            "software": None,
            "make": None,
            "model": None,
            "datetime": None,
            "lens_model": None,
            "ai_disclosed": False,
            "matched_tool": None,
            "raw_tags": {},
        }

        if not image_path.exists():
            return result

        raw_tags = {}

        # 1. First pass: ExifRead library
        import io, contextlib
        try:
            with open(image_path, "rb") as f:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    tags = exifread.process_file(f, details=False)
                for tag_name, val in tags.items():
                    # Filter out binary thumbnail/preview tags
                    if "JPEGThumbnail" not in tag_name and "MakerNote" not in tag_name:
                        raw_tags[tag_name] = str(val).strip()
        except Exception as e:
            logger.debug(f"ExifRead parsing error: {e}")

        # 2. Second pass: Pillow getexif()
        try:
            with Image.open(image_path) as img:
                exif_data = img.getexif()
                if exif_data:
                    for tag_id, val in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        if tag_name not in raw_tags and not isinstance(val, (bytes, bytearray)):
                            raw_tags[tag_name] = str(val).strip()

                # Also check PNG text chunks if format is PNG
                if hasattr(img, "info") and isinstance(img.info, dict):
                    for k, v in img.info.items():
                        if isinstance(v, str) and len(v) < 1000:
                            raw_tags[f"PNG:{k}"] = v.strip()
        except Exception as e:
            logger.debug(f"Pillow EXIF parsing error: {e}")

        if not raw_tags:
            result["exif_found"] = False
            return result

        result["exif_found"] = True
        result["raw_tags"] = raw_tags

        # Extract standard tags
        software_candidates = [
            raw_tags.get("Image Software"),
            raw_tags.get("Software"),
            raw_tags.get("Image ProcessingSoftware"),
            raw_tags.get("PNG:Software"),
            raw_tags.get("PNG:Comment"),
            raw_tags.get("PNG:Description"),
        ]
        result["software"] = next((s for s in software_candidates if s), None)

        result["make"] = raw_tags.get("Image Make") or raw_tags.get("Make")
        result["model"] = raw_tags.get("Image Model") or raw_tags.get("Model")
        result["datetime"] = (
            raw_tags.get("EXIF DateTimeOriginal")
            or raw_tags.get("Image DateTime")
            or raw_tags.get("DateTime")
        )
        result["lens_model"] = (
            raw_tags.get("EXIF LensModel")
            or raw_tags.get("LensModel")
        )

        # 3. Check for AI generation tool keywords across all text tags
        search_targets = [
            result["software"],
            raw_tags.get("Image Artist"),
            raw_tags.get("Image Copyright"),
            raw_tags.get("Image ImageDescription"),
            raw_tags.get("PNG:Comment"),
            raw_tags.get("PNG:Description"),
            raw_tags.get("PNG:prompt"),
            raw_tags.get("PNG:parameters"),  # Common Stable Diffusion WebUI chunk
        ]

        for target in search_targets:
            if target:
                match = match_ai_keyword(target)
                if match:
                    result["ai_disclosed"] = True
                    result["matched_tool"] = match[1]
                    break

        return result
