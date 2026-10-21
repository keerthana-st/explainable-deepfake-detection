"""
Data structures, schemas, and enumerations for the Secondary Region-Level Analysis System.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class RegionType(str, Enum):
    """Categorical types of segmentable regions."""
    FACE = "face"
    PERSON = "person"
    CLOTHING = "clothing"
    BACKGROUND = "background"
    OBJECT = "object"
    CUSTOM = "custom"


class RegionVerdict(str, Enum):
    """Verdict classification for an individual spatial region."""
    AI_EDITED = "AI-EDITED"
    REAL = "REAL"
    UNCERTAIN = "UNCERTAIN"
    INSUFFICIENT_SIZE = "INSUFFICIENT_SIZE"


class AgreementType(str, Enum):
    """Level of evidence consensus between Classifier and Frequency agents."""
    STRONG_AGREEMENT = "STRONG_AGREEMENT"
    WEAK_AGREEMENT = "WEAK_AGREEMENT"
    DISAGREEMENT = "DISAGREEMENT"
    INCONCLUSIVE = "INCONCLUSIVE"


class ConsistencyStatus(str, Enum):
    """Consistency outcome comparing Primary Global Verdict to Regional Evidence."""
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class SecondaryTriggerMode(str, Enum):
    """Configurable trigger policies for launching Secondary System."""
    ALWAYS = "ALWAYS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    DISAGREEMENT = "DISAGREEMENT"
    SUSPICIOUS_ONLY = "SUSPICIOUS_ONLY"
    MANUAL = "MANUAL"


@dataclass
class Region:
    """
    Representation of an isolated spatial region in an image.
    """
    region_id: str                      # e.g., 'face_01', 'clothing_01', 'background'
    region_type: RegionType             # Categorical type
    bbox: Tuple[int, int, int, int]     # [ymin, xmin, ymax, xmax] in original pixel coordinates
    mask: Optional[np.ndarray] = None   # Boolean 2D mask of image dimensions (H, W)
    area_pixels: int = 0
    confidence: float = 1.0             # Detection confidence of segmenter
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def height(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "region_type": self.region_type.value if isinstance(self.region_type, RegionType) else str(self.region_type),
            "bbox": list(self.bbox),
            "width": self.width,
            "height": self.height,
            "area_pixels": self.area_pixels,
            "confidence": round(float(self.confidence), 4),
            "metadata": self.metadata,
        }


@dataclass
class RegionEvidence:
    """
    Extracted forensic evidence and verdict for a specific region.
    """
    region_id: str = ""
    region_type: str = ""
    bbox: List[int] = field(default_factory=list)
    classifier_prediction: str = "REAL"         # 'REAL' or 'FAKE'
    real_probability: float = 1.0            # P(Real)
    fake_probability: float = 0.0            # P(Fake)
    frequency_score: float = 0.0             # [0.0, 1.0] frequency anomaly
    frequency_anomaly: bool = False          # True if frequency exceeds threshold
    agent_agreement: bool = True             # True if both agents agree
    agreement_type: Any = "STRONG_AGREEMENT" # AgreementType enum value or str
    combined_score: float = 0.0              # Weighted suspiciousness score [0.0, 1.0]
    region_verdict: Any = "REAL"             # 'AI-EDITED', 'REAL', 'UNCERTAIN' or RegionVerdict
    confidence: Any = "HIGH"                 # 'HIGH', 'MEDIUM', 'LOW' or float
    explanation: str = ""                    # Human-readable regional forensic reasoning
    gradcam_path: Optional[str] = None       # Path to selective Grad-CAM overlay
    crop_path: Optional[str] = None          # Path to cropped region image
    heatmap_path: Optional[str] = None       # Path to regional heatmap
    region: Optional[Region] = None
    classifier_score: Optional[float] = None

    def __post_init__(self):
        if self.region is not None:
            if not self.region_id:
                self.region_id = self.region.region_id
            if not self.region_type:
                self.region_type = self.region.region_type.value if hasattr(self.region.region_type, "value") else str(self.region.region_type)
            if not self.bbox:
                self.bbox = list(self.region.bbox)
        if self.classifier_score is not None:
            self.fake_probability = self.classifier_score
            self.real_probability = 1.0 - self.classifier_score
        if hasattr(self.agreement_type, "value"):
            self.agreement_type = self.agreement_type.value
        if hasattr(self.region_verdict, "value"):
            self.region_verdict = self.region_verdict.value
        if isinstance(self.confidence, (int, float)):
            self.confidence = "HIGH" if self.confidence >= 0.7 else ("LOW" if self.confidence <= 0.4 else "MEDIUM")



    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "region_type": self.region_type,
            "bbox": self.bbox,
            "classifier_prediction": self.classifier_prediction,
            "real_probability": round(float(self.real_probability), 4),
            "fake_probability": round(float(self.fake_probability), 4),
            "classifier_score": round(float(self.fake_probability), 4),
            "frequency_score": round(float(self.frequency_score), 4),
            "frequency_anomaly": bool(self.frequency_anomaly),
            "agent_agreement": bool(self.agent_agreement),
            "agreement_type": self.agreement_type,
            "agreement": self.agreement_type,
            "combined_score": round(float(self.combined_score), 4),
            "region_verdict": self.region_verdict,
            "verdict": self.region_verdict,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "gradcam_path": self.gradcam_path,
            "crop_path": self.crop_path,
            "heatmap_path": self.heatmap_path,
        }



@dataclass
class RegionalConsistencyResult:
    """
    Forensic comparison between Primary Global Verdict and Secondary Regional Evidence.
    """
    status: ConsistencyStatus
    primary_verdict: str
    explanation: str
    suspicious_regions: List[str] = field(default_factory=list)
    authentic_regions: List[str] = field(default_factory=list)
    requires_review: bool = False
    inconsistency_warning: Optional[str] = None

    @property
    def warning(self) -> Optional[str]:
        return self.inconsistency_warning

    def to_dict(self) -> Dict[str, Any]:

        return {
            "status": self.status.value if isinstance(self.status, ConsistencyStatus) else str(self.status),
            "primary_verdict": self.primary_verdict,
            "explanation": self.explanation,
            "suspicious_regions": self.suspicious_regions,
            "authentic_regions": self.authentic_regions,
            "requires_review": self.requires_review,
            "inconsistency_warning": self.inconsistency_warning,
        }


@dataclass
class SecondaryAnalysisResult:
    """
    Unified output of the Secondary System.
    """
    enabled: bool
    trigger_mode: str
    trigger_fired: bool
    regions: List[RegionEvidence] = field(default_factory=list)
    regional_consistency: Optional[RegionalConsistencyResult] = None
    suspicious_regions: List[str] = field(default_factory=list)
    visualizations: Dict[str, str] = field(default_factory=dict) # overlay, heatmap, gradcam paths
    summary_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "trigger_mode": self.trigger_mode,
            "trigger_fired": self.trigger_fired,
            "executed": self.trigger_fired,
            "region_count": len(self.regions),
            "suspicious_region_count": len(self.suspicious_regions),
            "regions": [r.to_dict() for r in self.regions],
            "regional_consistency": self.regional_consistency.to_dict() if self.regional_consistency else None,
            "suspicious_regions": self.suspicious_regions,
            "visualizations": self.visualizations,
            "summary_explanation": self.summary_explanation,
        }

