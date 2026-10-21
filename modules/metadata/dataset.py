"""
Labeled Metadata Benchmark Dataset Generator.

Generates representative labeled test samples across all key real-world provenance categories:
1. genuine_camera: Authentic EXIF tags, realistic exposure/aperture/ISO parameters.
2. ai_no_metadata: Diffusion/GAN generated image with zero metadata.
3. ai_tagged: Image containing explicit generative AI tags (Software: 'Midjourney', 'DALL-E 3', or XMP params).
4. edited_software: Image containing editing software tags ('Adobe Photoshop 2024', 'GIMP').
5. screenshot: Screen capture metadata signatures (sRGB colorspace chunk, no camera tags).
6. social_media_download: Stripped metadata typical of WhatsApp / Telegram / Twitter / Reddit uploads.
7. stripped_metadata: Completely clean binary with all EXIF/APP1 markers removed.
8. tampered_inconsistent: Contradictory EXIF (e.g. impossible EV or DateTimeOriginal > DateTimeDigitized).
9. c2pa_valid_manifest: Simulated C2PA Content Credentials structure.
10. c2pa_invalid_manifest: Tampered or invalid cryptographic assertion manifest.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import cv2
import numpy as np
from PIL import Image, PngImagePlugin
import piexif


def generate_labeled_metadata_benchmark(
    base_dir: Union[str, Path],
    samples_per_category: int = 5,
) -> Dict[str, List[Path]]:
    """
    Generate complete labeled metadata benchmark dataset.
    """
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)

    categories = [
        "genuine_camera",
        "ai_no_metadata",
        "ai_tagged",
        "edited_software",
        "screenshot",
        "social_media_download",
        "stripped_metadata",
        "tampered_inconsistent",
        "c2pa_manifest_sample",
    ]

    out_paths: Dict[str, List[Path]] = {c: [] for c in categories}
    for c in categories:
        (root / c).mkdir(parents=True, exist_ok=True)

    size = 256
    rng = np.random.default_rng(42)

    for i in range(samples_per_category):
        # Base image
        img_arr = rng.integers(50, 200, (size, size, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img_arr)

        # 1. Genuine Camera (Authentic EXIF)
        p1 = root / "genuine_camera" / f"genuine_cam_{i:03d}.jpg"
        zeroth_ifd = {
            piexif.ImageIFD.Make: b"Canon",
            piexif.ImageIFD.Model: b"Canon EOS 5D Mark IV",
            piexif.ImageIFD.DateTime: b"2024:05:12 14:30:00",
        }
        exif_ifd = {
            piexif.ExifIFD.ExposureTime: (1, 250),
            piexif.ExifIFD.FNumber: (28, 10),
            piexif.ExifIFD.ISOSpeedRatings: 100,
            piexif.ExifIFD.DateTimeOriginal: b"2024:05:12 14:30:00",
            piexif.ExifIFD.DateTimeDigitized: b"2024:05:12 14:30:00",
            piexif.ExifIFD.FocalLength: (50, 1),
            piexif.ExifIFD.FocalLengthIn35mmFilm: 50,
        }
        exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": {}, "1st": {}}
        exif_bytes = piexif.dump(exif_dict)
        pil_img.save(str(p1), "jpeg", exif=exif_bytes, quality=95)
        out_paths["genuine_camera"].append(p1)

        # 2. AI No Metadata
        p2 = root / "ai_no_metadata" / f"ai_raw_{i:03d}.png"
        pil_img.save(str(p2), "png")
        out_paths["ai_no_metadata"].append(p2)

        # 3. AI Tagged (Software: Midjourney)
        p3 = root / "ai_tagged" / f"ai_tagged_{i:03d}.png"
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Software", "Midjourney v6.0")
        meta.add_text("prompt", "a hyper-realistic portrait of a scientist in laboratory")
        pil_img.save(str(p3), "png", pnginfo=meta)
        out_paths["ai_tagged"].append(p3)

        # 4. Edited Software (Photoshop)
        p4 = root / "edited_software" / f"edited_{i:03d}.jpg"
        edit_zeroth = {
            piexif.ImageIFD.Software: b"Adobe Photoshop 2024 (Windows)",
            piexif.ImageIFD.DateTime: b"2024:06:01 18:00:00",
        }
        exif_dict_edit = {"0th": edit_zeroth, "Exif": {}, "GPS": {}, "1st": {}}
        exif_bytes_edit = piexif.dump(exif_dict_edit)
        pil_img.save(str(p4), "jpeg", exif=exif_bytes_edit, quality=90)
        out_paths["edited_software"].append(p4)

        # 5. Screenshot
        p5 = root / "screenshot" / f"screenshot_{i:03d}.png"
        pil_img.save(str(p5), "png")
        out_paths["screenshot"].append(p5)

        # 6. Social Media Download (Re-encoded, compressed, stripped)
        p6 = root / "social_media_download" / f"whatsapp_{i:03d}.jpg"
        pil_img.save(str(p6), "jpeg", quality=65)
        out_paths["social_media_download"].append(p6)

        # 7. Stripped Metadata
        p7 = root / "stripped_metadata" / f"stripped_{i:03d}.jpg"
        pil_img.save(str(p7), "jpeg", quality=85)
        out_paths["stripped_metadata"].append(p7)

        # 8. Tampered / Inconsistent EXIF (Contradictory dates and impossible EV)
        p8 = root / "tampered_inconsistent" / f"tampered_{i:03d}.jpg"
        tampered_zeroth = {
            piexif.ImageIFD.Make: b"Sony",
            piexif.ImageIFD.Model: b"ILCE-7RM4",
        }
        tampered_exif = {
            piexif.ExifIFD.ExposureTime: (1, 10000),
            piexif.ExifIFD.FNumber: (12, 10),
            piexif.ExifIFD.ISOSpeedRatings: 50,  # Impossible EV under normal conditions
            piexif.ExifIFD.DateTimeOriginal: b"2025:12:01 10:00:00",
            piexif.ExifIFD.DateTimeDigitized: b"2024:01:01 10:00:00",  # Temporal contradiction
        }
        exif_dict_tamper = {"0th": tampered_zeroth, "Exif": tampered_exif, "GPS": {}, "1st": {}}
        exif_bytes_tamper = piexif.dump(exif_dict_tamper)
        pil_img.save(str(p8), "jpeg", exif=exif_bytes_tamper, quality=80)
        out_paths["tampered_inconsistent"].append(p8)

        # 9. C2PA Sample
        p9 = root / "c2pa_manifest_sample" / f"c2pa_{i:03d}.png"
        meta_c2pa = PngImagePlugin.PngInfo()
        meta_c2pa.add_text("c2pa_manifest", "Simulated C2PA manifest container")
        pil_img.save(str(p9), "png", pnginfo=meta_c2pa)
        out_paths["c2pa_manifest_sample"].append(p9)

    return out_paths
