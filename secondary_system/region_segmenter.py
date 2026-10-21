"""
Modular Region Segmentation for the Secondary System.
Identifies and segments Faces, Person silhouettes, Clothing (approximated),
Background, and Non-human Objects.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .models import Region, RegionType


class BaseSegmenter(ABC):
    """Abstract base class for modular region segmenters."""

    @abstractmethod
    def segment(self, np_img: np.ndarray, context: Optional[Dict[str, Any]] = None) -> List[Region]:
        """
        Segment designated semantic regions from an RGB numpy image.

        Args:
            np_img: RGB image array of shape (H, W, 3)
            context: Optional dictionary containing upstream segmentation masks/context

        Returns:
            List of detected Region objects
        """
        pass


class FaceSegmenter(BaseSegmenter):
    """
    Detects and segments individual human faces independently.
    Supports multi-face images (Face 1, Face 2, Face 3).
    """

    def __init__(self, min_size: Tuple[int, int] = (20, 20)):
        self.min_size = min_size
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        # Profile face cascade for profile angles
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        self.profile_cascade = cv2.CascadeClassifier(profile_path) if cv2.os.path.exists(profile_path) else None

    def segment(self, np_img: np.ndarray, context: Optional[Dict[str, Any]] = None) -> List[Region]:
        h, w = np_img.shape[:2]
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        # Multi-scale detection
        faces = list(self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        ))

        # Check profile if no frontal face was detected
        if len(faces) == 0 and self.profile_cascade is not None:
            faces = list(self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=self.min_size
            ))

        # Sort faces left-to-right
        faces = sorted(faces, key=lambda f: f[0])

        regions: List[Region] = []
        for idx, (fx, fy, fw, fh) in enumerate(faces):
            region_id = f"face_{idx + 1:02d}"

            # Expand bbox slightly to encompass hairline and chin (15% padding)
            pad_x = int(fw * 0.12)
            pad_y = int(fh * 0.15)
            xmin = max(0, fx - pad_x)
            xmax = min(w, fx + fw + pad_x)
            ymin = max(0, fy - pad_y)
            ymax = min(h, fy + fh + pad_y)

            # Create elliptical binary mask
            mask = np.zeros((h, w), dtype=bool)
            center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
            axes = (int((xmax - xmin) / 2), int((ymax - ymin) / 2))
            cv2.ellipse(
                mask.view(np.uint8),
                center=center,
                axes=axes,
                angle=0,
                startAngle=0,
                endAngle=360,
                color=1,
                thickness=-1,
            )

            area = int(np.sum(mask))
            regions.append(Region(
                region_id=region_id,
                region_type=RegionType.FACE,
                bbox=(ymin, xmin, ymax, xmax),
                mask=mask,
                area_pixels=area,
                confidence=0.95,
                metadata={"face_index": idx + 1, "raw_rect": [fx, fy, fw, fh]},
            ))

        return regions


class PersonSegmenter(BaseSegmenter):
    """
    Segments full human body / silhouette.
    Utilizes GrabCut seeded with detected face and central body priors,
    ensuring 100% offline local reliability without remote model downloads.
    """

    def segment(self, np_img: np.ndarray, context: Optional[Dict[str, Any]] = None) -> List[Region]:
        h, w = np_img.shape[:2]
        face_regions: List[Region] = context.get("face_regions", []) if context else []

        if not face_regions:
            # Check OpenCV HOG people detector
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            boxes, _ = hog.detectMultiScale(np_img, winStride=(8, 8), padding=(8, 8), scale=1.05)
            if len(boxes) == 0:
                return []
            bx, by, bw, bh = boxes[0]
            face_seed = (by, bx, by + int(bh * 0.25), bx + bw)
        else:
            face_seed = face_regions[0].bbox

        # Build GrabCut mask from face prior
        f_ymin, f_xmin, f_ymax, f_xmax = face_seed
        fw = f_xmax - f_xmin
        fh = f_ymax - f_ymin

        # Approximate human torso and body bounds below face
        body_ymin = max(0, f_ymin)
        body_ymax = min(h, f_ymax + int(fh * 4.5))
        body_xmin = max(0, f_xmin - int(fw * 1.5))
        body_xmax = min(w, f_xmax + int(fw * 1.5))

        if body_ymax <= body_ymin or body_xmax <= body_xmin:
            return []

        # Run GrabCut on bounding rect
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (body_xmin, body_ymin, body_xmax - body_xmin, body_ymax - body_ymin)

        try:
            cv2.grabCut(np_img, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
            person_mask = np.where((mask == 2) | (mask == 0), False, True)
        except Exception:
            # Fallback to rectangular torso mask if grabcut encounters boundary glitch
            person_mask = np.zeros((h, w), dtype=bool)
            person_mask[body_ymin:body_ymax, body_xmin:body_xmax] = True

        # Always ensure face area is included in person mask
        for f in face_regions:
            if f.mask is not None:
                person_mask = person_mask | f.mask

        area = int(np.sum(person_mask))
        if area < 400:
            return []

        # Find enclosing bbox
        y_indices, x_indices = np.where(person_mask)
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))

        return [Region(
            region_id="person_01",
            region_type=RegionType.PERSON,
            bbox=(ymin, xmin, ymax, xmax),
            mask=person_mask,
            area_pixels=area,
            confidence=0.88,
            metadata={"detection_source": "grabcut_face_prior"},
        )]


class ClothingSegmenter(BaseSegmenter):
    """
    Approximates clothing / body attire region by subtracting face masks from person mask.
    Documented explicitly as an approximation.
    """

    def segment(self, np_img: np.ndarray, context: Optional[Dict[str, Any]] = None) -> List[Region]:
        if not context:
            return []

        person_regions: List[Region] = context.get("person_regions", [])
        face_regions: List[Region] = context.get("face_regions", [])

        if not person_regions:
            return []

        person = person_regions[0]
        if person.mask is None:
            return []

        # Clothing = Person Mask MINUS all Face Masks
        clothing_mask = person.mask.copy()
        for f in face_regions:
            if f.mask is not None:
                clothing_mask = clothing_mask & (~f.mask)

        area = int(np.sum(clothing_mask))
        if area < 300:
            return []

        y_indices, x_indices = np.where(clothing_mask)
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))

        return [Region(
            region_id="clothing_01",
            region_type=RegionType.CLOTHING,
            bbox=(ymin, xmin, ymax, xmax),
            mask=clothing_mask,
            area_pixels=area,
            confidence=0.82,
            metadata={"approximation": "person_silhouette_minus_face_mask"},
        )]


class BackgroundSegmenter(BaseSegmenter):
    """
    Extracts authentic background by taking the entire image frame minus all foreground masks.
    Preserves full spatial background geometry.
    """

    def segment(self, np_img: np.ndarray, context: Optional[Dict[str, Any]] = None) -> List[Region]:
        h, w = np_img.shape[:2]
        foreground_regions: List[Region] = context.get("foreground_regions", []) if context else []

        # Full image mask
        background_mask = np.ones((h, w), dtype=bool)
        for fg in foreground_regions:
            if fg.mask is not None:
                background_mask = background_mask & (~fg.mask)

        area = int(np.sum(background_mask))
        if area < 200:
            # If foreground occupies 100%, background is perimeter border
            background_mask = np.ones((h, w), dtype=bool)
            background_mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = False
            area = int(np.sum(background_mask))

        return [Region(
            region_id="background",
            region_type=RegionType.BACKGROUND,
            bbox=(0, 0, h, w),
            mask=background_mask,
            area_pixels=area,
            confidence=0.90,
            metadata={"source": "inverse_foreground"},
        )]


class ObjectSegmenter(BaseSegmenter):
    """
    Fallback segmenter for non-human images (e.g. dog.jpg, animals, cars, synthetic objects).
    Extracts salient central foreground subject vs surrounding background.
    """

    def segment(self, np_img: np.ndarray, context: Optional[Dict[str, Any]] = None) -> List[Region]:
        h, w = np_img.shape[:2]

        # Use Otsu thresholding on gradient magnitude / color variance to locate primary subject
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest)
            if bw >= 20 and bh >= 20 and (bw * bh) >= (w * h * 0.05):
                obj_mask = np.zeros((h, w), dtype=bool)
                cv2.drawContours(obj_mask.view(np.uint8), [largest], -1, 1, thickness=-1)
                ymin, xmin, ymax, xmax = by, bx, by + bh, bx + bw
                area = int(np.sum(obj_mask))
                return [Region(
                    region_id="subject_01",
                    region_type=RegionType.OBJECT,
                    bbox=(ymin, xmin, ymax, xmax),
                    mask=obj_mask,
                    area_pixels=area,
                    confidence=0.85,
                    metadata={"source": "salient_contour"},
                )]

        # Default central subject crop (60% center)
        ymin, ymax = int(h * 0.20), int(h * 0.80)
        xmin, xmax = int(w * 0.20), int(w * 0.80)
        obj_mask = np.zeros((h, w), dtype=bool)
        obj_mask[ymin:ymax, xmin:xmax] = True
        return [Region(
            region_id="subject_01",
            region_type=RegionType.OBJECT,
            bbox=(ymin, xmin, ymax, xmax),
            mask=obj_mask,
            area_pixels=int(np.sum(obj_mask)),
            confidence=0.75,
            metadata={"source": "center_prior"},
        )]


class UnifiedRegionSegmenter:
    """
    Master segmentation orchestrator coordinating Face, Person, Clothing,
    Background, and Object segmenters into a unified, clean list of Region objects.
    """

    def __init__(self):
        self.face_segmenter = FaceSegmenter()
        self.person_segmenter = PersonSegmenter()
        self.clothing_segmenter = ClothingSegmenter()
        self.background_segmenter = BackgroundSegmenter()
        self.object_segmenter = ObjectSegmenter()

    def segment_image(self, np_img: np.ndarray) -> List[Region]:
        """
        Segment an image into all available semantic regions.
        Never crashes; gracefully falls back to available regions.
        """
        h, w = np_img.shape[:2]
        all_regions: List[Region] = []

        # 1. Detect Faces
        face_regions = self.face_segmenter.segment(np_img)
        all_regions.extend(face_regions)

        # 2. Detect Person / Silhouette
        person_regions = self.person_segmenter.segment(np_img, context={"face_regions": face_regions})

        # 3. If Person exists, segment Clothing
        if person_regions:
            clothing_regions = self.clothing_segmenter.segment(
                np_img, context={"person_regions": person_regions, "face_regions": face_regions}
            )
            all_regions.extend(clothing_regions)

        # 4. If no human faces or person detected, use Object fallback
        if not face_regions and not person_regions:
            object_regions = self.object_segmenter.segment(np_img)
            all_regions.extend(object_regions)

        # 5. Extract Background (Image minus all foreground regions)
        foreground = [r for r in all_regions if r.region_type != RegionType.BACKGROUND]
        bg_regions = self.background_segmenter.segment(np_img, context={"foreground_regions": foreground})
        all_regions.extend(bg_regions)

        return all_regions

    # Convenience alias
    segment = segment_image


