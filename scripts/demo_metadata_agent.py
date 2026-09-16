"""
Interactive Demonstration CLI for Module 3: Metadata Agent.
Demonstrates all 4 distinct metadata classification states and provenance extraction.
"""

from pathlib import Path
import tempfile
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ExifTags

from modules.metadata.agent import MetadataAgent
from modules.metadata.result import MetadataState


def create_synthetic_camera_photo(output_path: Path) -> Path:
    """Creates a test JPEG with genuine camera hardware EXIF tags (Canon EOS R5)."""
    img = Image.new("RGB", (300, 300), color=(140, 180, 220))
    exif = img.getexif()
    exif[0x010F] = "Canon"                                     # Make
    exif[0x0110] = "Canon EOS R5"                             # Model
    exif[0x0131] = "Digital Photo Professional 4.14"          # Software
    exif[0x0132] = "2025:10:14 11:42:05"                      # DateTime
    exif[0xA434] = "RF24-70mm F2.8 L IS USM"                  # LensModel
    img.save(output_path, format="JPEG", exif=exif)
    return output_path


def create_stripped_social_media_image(output_path: Path) -> Path:
    """Creates a completely clean image without any EXIF/XMP/C2PA metadata."""
    img = Image.new("RGB", (300, 300), color=(200, 200, 200))
    img.save(output_path, format="PNG")
    return output_path


def create_exif_ai_disclosed_image(output_path: Path) -> Path:
    """Creates an image with Midjourney / Stable Diffusion disclosed in EXIF."""
    img = Image.new("RGB", (300, 300), color=(120, 100, 160))
    exif = img.getexif()
    exif[0x0131] = "Midjourney v6.0 / Diffusion Core"         # Software
    exif[0x010E] = "A portrait of an astronaut generated via Midjourney"  # ImageDescription
    img.save(output_path, format="JPEG", exif=exif)
    return output_path


def main():
    print("=" * 80)
    print("       MODULE 3: METADATA AGENT — COMPREHENSIVE 4-STATE AUDIT DEMO")
    print("=" * 80)
    print(MetadataAgent.PROVENANCE_SCOPE_NOTICE)
    print("=" * 80)

    agent = MetadataAgent()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # State 1: ai_disclosed (C2PA and EXIF tests)
        print("\n" + "#" * 80)
        print("STATE 1: 'ai_disclosed'")
        print("#" * 80)

        # Real C2PA image if exists
        real_c2pa = Path(r"D:\ChatGPT Image Sep 3, 2026, 03_02_44 PM.png")
        if real_c2pa.exists():
            print(f"\n[Test 1A: Real C2PA Image from ChatGPT ({real_c2pa.name})]")
            res_c2pa = agent.analyze(real_c2pa)
            print(f"  • Flag:           {res_c2pa.metadata_flag.upper()}")
            print(f"  • Disclosed Tool: {res_c2pa.disclosed_tool}")
            print(f"  • Data Source:    {res_c2pa.data_source_tag}")
            print(f"  • Signer:         {res_c2pa.c2pa_details.signing_issuer}")
            print(f"  • Timestamp:      {res_c2pa.c2pa_details.timestamp}")
            print(f"  • Reasoning:\n    {res_c2pa.reasoning}")

        exif_ai_img = create_exif_ai_disclosed_image(tmp_path / "midjourney_disclosed.jpg")
        print(f"\n[Test 1B: Image with Disclosed AI Tool in EXIF ({exif_ai_img.name})]")
        res_exif = agent.analyze(exif_ai_img)
        print(f"  • Flag:           {res_exif.metadata_flag.upper()}")
        print(f"  • Disclosed Tool: {res_exif.disclosed_tool}")
        print(f"  • Data Source:    {res_exif.data_source_tag}")
        print(f"  • Reasoning:\n    {res_exif.reasoning}")

        # State 2: manifest_present_generator_unclear
        print("\n" + "#" * 80)
        print("STATE 2: 'manifest_present_generator_unclear'")
        print("#" * 80)
        # Construct a PNG file with a raw caBX box containing generic JUMBF data without known tool name
        unclear_png = tmp_path / "unclear_manifest.png"
        raw_png = create_stripped_social_media_image(tmp_path / "base.png")
        with open(raw_png, "rb") as f_in:
            data = f_in.read()
        # Insert a dummy caBX chunk right before IEND
        iend_pos = data.find(b"IEND")
        dummy_cabx_payload = b"jumdc2pa\x00\x00urn:c2pa:generic-custom-workflow-token"
        import struct, zlib
        chunk_len = struct.pack(">I", len(dummy_cabx_payload))
        chunk_type = b"caBX"
        crc = struct.pack(">I", zlib.crc32(chunk_type + dummy_cabx_payload) & 0xffffffff)
        modified_data = data[:iend_pos - 4] + chunk_len + chunk_type + dummy_cabx_payload + crc + data[iend_pos - 4:]
        with open(unclear_png, "wb") as f_out:
            f_out.write(modified_data)

        res_unclear = agent.analyze(unclear_png)
        print(f"[Test 2: Manifest Present but Generator Unclear ({unclear_png.name})]")
        print(f"  • Flag:           {res_unclear.metadata_flag.upper()}")
        print(f"  • Disclosed Tool: {res_unclear.disclosed_tool}")
        print(f"  • Data Source:    {res_unclear.data_source_tag}")
        print(f"  • Reasoning:\n    {res_unclear.reasoning}")

        # State 3: present_no_ai_signal
        print("\n" + "#" * 80)
        print("STATE 3: 'present_no_ai_signal'")
        print("#" * 80)
        cam_img = create_synthetic_camera_photo(tmp_path / "authentic_canon_r5.jpg")
        res_cam = agent.analyze(cam_img)
        print(f"[Test 3: Authentic Camera Photo with Full EXIF ({cam_img.name})]")
        print(f"  • Flag:           {res_cam.metadata_flag.upper()}")
        print(f"  • Camera Make:    {res_cam.exif_summary.get('make')}")
        print(f"  • Camera Model:   {res_cam.exif_summary.get('model')}")
        print(f"  • Data Source:    {res_cam.data_source_tag}")
        print(f"  • Reasoning:\n    {res_cam.reasoning}")

        # State 4: missing
        print("\n" + "#" * 80)
        print("STATE 4: 'missing'")
        print("#" * 80)
        stripped_img = create_stripped_social_media_image(tmp_path / "stripped_social_media.png")
        res_missing = agent.analyze(stripped_img)
        print(f"[Test 4: Clean / Stripped Social Media Image ({stripped_img.name})]")
        print(f"  • Flag:           {res_missing.metadata_flag.upper()}")
        print(f"  • Disclosed Tool: {res_missing.disclosed_tool}")
        print(f"  • Data Source:    {res_missing.data_source_tag}")
        print(f"  • Reasoning:\n    {res_missing.reasoning}")

        # Error Handling: Corrupted File
        print("\n" + "#" * 80)
        print("TEST 5: Corrupted / Malformed File Handling")
        print("#" * 80)
        corrupt_file = tmp_path / "corrupted_garbage.jpg"
        with open(corrupt_file, "wb") as f:
            f.write(b"\xFF\xD8\xFF\xE1\x00\x10GARBAGE_EXIF_CORRUPTED_HEADER_DATA_1234567890")
        res_corrupt = agent.analyze(corrupt_file)
        print(f"[Test 5: Deliberately Corrupted Header ({corrupt_file.name})]")
        print(f"  • Flag:           {res_corrupt.metadata_flag.upper()}")
        print(f"  • Data Source:    {res_corrupt.data_source_tag}")
        print(f"  • Status:         Handled gracefully without pipeline crash!")

    print("\n" + "=" * 80)
    print("Metadata Agent 4-State Verification Complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
