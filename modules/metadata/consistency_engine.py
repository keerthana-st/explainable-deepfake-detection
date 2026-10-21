"""
Optical and Physical Consistency Engine for Metadata Agent.

Applies deterministic physical laws and camera sensor heuristics to detect
forged, tampered, or physically impossible EXIF metadata:
1. Exposure Reciprocity & EV Calculation (Aperture, Shutter Speed, ISO).
2. Focal Length vs. 35mm Equivalent vs. Sensor Crop Factor consistency.
3. Timestamp Sequence (DateTimeOriginal <= DateTimeDigitized <= DateTime).
4. Editing Software Detection (Photoshop, GIMP, Lightroom, Canva).
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class OpticalConsistencyReport:
    """Audit report of physical and optical metadata consistency."""

    is_consistent: bool
    contradictions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    calculated_ev: Optional[float] = None
    is_edited_software: bool = False
    detected_software: Optional[str] = None
    optical_details: Dict[str, Any] = field(default_factory=dict)


class MetadataConsistencyEngine:
    """
    Evaluates optical, physical, and temporal consistency of EXIF tags.
    """

    KNOWN_EDITING_SOFTWARE = [
        "photoshop",
        "gimp",
        "lightroom",
        "canva",
        "pixlr",
        "paint.net",
        "snapseed",
        "picsart",
        "imagemagick",
    ]

    def verify_consistency(self, raw_tags: Dict[str, Any]) -> OpticalConsistencyReport:
        contradictions = []
        warnings = []
        details = {}
        is_edited = False
        detected_soft = None

        if not raw_tags:
            return OpticalConsistencyReport(
                is_consistent=True,
                warnings=["No EXIF tags present to verify optical consistency."],
            )

        # 1. Check Software Tags for Editing Tools
        software_str = str(raw_tags.get("Image Software", "") or raw_tags.get("Software", "")).lower()
        for soft in self.KNOWN_EDITING_SOFTWARE:
            if soft in software_str:
                is_edited = True
                detected_soft = software_str.strip()
                warnings.append(f"Image processed with editing software: '{detected_soft}'.")
                break

        # 2. Exposure Reciprocity & EV Check: EV = log2(N^2 / t) - log2(S / 100)
        f_number = self._extract_float(raw_tags.get("EXIF FNumber"))
        exp_time = self._extract_float(raw_tags.get("EXIF ExposureTime"))
        iso = self._extract_float(raw_tags.get("EXIF ISOSpeedRatings")) or self._extract_float(raw_tags.get("EXIF PhotographicSensitivity"))

        ev = None
        if f_number is not None and exp_time is not None and iso is not None:
            if f_number > 0 and exp_time > 0 and iso > 0:
                try:
                    # EV at ISO 100: log2(N^2 / t)
                    ev_100 = math.log2((f_number**2) / exp_time)
                    ev = ev_100 - math.log2(iso / 100.0)
                    details["calculated_ev"] = round(ev, 2)
                    details["f_number"] = f_number
                    details["exposure_time"] = exp_time
                    details["iso"] = iso

                    # Physical bounds check for photography: EV usually in [-6, 20]
                    if ev < -8.0 or ev > 24.0:
                        contradictions.append(
                            f"Physically impossible exposure value (EV={ev:.1f}): "
                            f"f/{f_number}, {exp_time}s, ISO {iso} does not correspond to natural optical settings."
                        )
                except (ValueError, ZeroDivisionError):
                    pass

        # 3. Focal Length vs. 35mm Equivalent
        focal_len = self._extract_float(raw_tags.get("EXIF FocalLength"))
        focal_len_35 = self._extract_float(raw_tags.get("EXIF FocalLengthIn35mmFilm"))

        if focal_len is not None and focal_len_35 is not None:
            details["focal_length"] = focal_len
            details["focal_length_35mm"] = focal_len_35
            if focal_len > 0 and focal_len_35 > 0:
                crop_factor = focal_len_35 / focal_len
                details["implied_crop_factor"] = round(crop_factor, 2)
                # Sensor crop factor standard range: ~0.8 (medium format) to ~8.0 (small phone sensor)
                if crop_factor < 0.5 or crop_factor > 12.0:
                    contradictions.append(
                        f"Optical impossibility: Focal length {focal_len}mm with 35mm equivalent {focal_len_35}mm "
                        f"implies sensor crop factor of {crop_factor:.1f}x (outside physical camera sensor bounds)."
                    )

        # 4. Temporal Sequence Consistency
        orig_time = self._parse_exif_datetime(raw_tags.get("EXIF DateTimeOriginal"))
        dig_time = self._parse_exif_datetime(raw_tags.get("EXIF DateTimeDigitized"))
        mod_time = self._parse_exif_datetime(raw_tags.get("Image DateTime"))

        if orig_time and dig_time and orig_time > dig_time:
            contradictions.append(
                f"Temporal anomaly: DateTimeOriginal ({orig_time}) is later than DateTimeDigitized ({dig_time})."
            )

        if orig_time and mod_time and orig_time > mod_time:
            contradictions.append(
                f"Temporal anomaly: DateTimeOriginal ({orig_time}) is later than file modification DateTime ({mod_time})."
            )

        is_consistent = len(contradictions) == 0

        return OpticalConsistencyReport(
            is_consistent=is_consistent,
            contradictions=contradictions,
            warnings=warnings,
            calculated_ev=ev,
            is_edited_software=is_edited,
            detected_software=detected_soft,
            optical_details=details,
        )

    def _extract_float(self, val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            val_str = str(val).strip()
            if "/" in val_str:
                num, den = val_str.split("/")
                return float(num) / float(den)
            return float(val_str)
        except (ValueError, ZeroDivisionError):
            return None

    def _parse_exif_datetime(self, val: Any) -> Optional[datetime]:
        if val is None:
            return None
        s = str(val).strip()
        for fmt in ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None
