"""
Common Image Preprocessing Module - Standalone Entry Point & CLI Interface.

Usage:
    python preprocessing.py "path/to/image.jpg"
    python preprocessing.py "path/to/image.png" --size 512 512 --norm imagenet
    python preprocessing.py "path/to/folder"
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.preprocessing import (
    ImagePreprocessor,
    PreprocessingConfig,
    PreprocessingResult,
    preprocess_image,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMATS,
)


def main():
    parser = argparse.ArgumentParser(
        description="Explainable Deepfake Detection - Common Image Preprocessing Module"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an image file or directory containing images (JPG, JPEG, PNG)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("WIDTH", "HEIGHT"),
        help="Target bounding box resolution (default: 512 512)",
    )
    parser.add_argument(
        "--resize-method",
        type=str,
        default="aspect_ratio_preserve",
        choices=["aspect_ratio_preserve", "pad_square", "direct", "none"],
        help="Resolution adaptation method (default: aspect_ratio_preserve)",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="imagenet",
        choices=["imagenet", "zero_to_one", "minus_one_to_one", "none"],
        help="Normalization strategy for deep vision classifier (default: imagenet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/preprocessing",
        help="Directory to persist decoupled representations and metadata",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable disk persistence of decoupled representations",
    )

    args = parser.parse_args()
    target_p = Path(args.input_path)

    if not target_p.exists():
        print(f"\nError: Specified input path does not exist: {target_p}")
        sys.exit(1)

    cfg = PreprocessingConfig(
        target_size=tuple(args.size) if args.size else None,
        resize_method=args.resize_method,
        normalization_method=args.norm,
        output_dir=args.output_dir if not args.no_save else None,
        save_artifacts=not args.no_save,
    )
    preprocessor = ImagePreprocessor(config=cfg)

    # Collect images
    images_to_process = []
    if target_p.is_file():
        images_to_process.append(target_p)
    elif target_p.is_dir():
        for f in sorted(target_p.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                images_to_process.append(f)
        if not images_to_process:
            print(f"\nNo supported images found in directory: {target_p}")
            sys.exit(1)

    print(f"\nProcessing {len(images_to_process)} image(s)...")
    for img_path in images_to_process:
        result: PreprocessingResult = preprocessor.preprocess(img_path)
        print("\n" + result.format_report())
        if result.is_success and not args.no_save and result.saved_paths:
            print("\nPersisted Forensic Representations:")
            for k, p in result.saved_paths.items():
                print(f"  • {k.capitalize():<12}: {p}")


if __name__ == "__main__":
    main()
