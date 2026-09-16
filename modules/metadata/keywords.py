"""
Maintained Registry of AI Generation and Manipulation Tool Keywords.

This list is intentionally maintained as a clean, transparent, and editable
configuration rather than embedded inside procedural parsing logic.

MAINTENANCE & UPDATE INSTRUCTIONS:
==================================
As new generative AI models and image synthesis platforms launch, add their
canonical names, identifiers, and common spelling variations to AI_TOOL_KEYWORDS.

NOTE ON COVERAGE LIMITATION:
This keyword dictionary is deterministic and rule-based. It requires periodic
manual updates and cannot automatically infer or detect undisclosed or newly
released models whose identifiers have not yet been registered.
"""

from typing import Optional, List, Tuple

# Maintained list of known generative AI tools, models, engines, and services.
# Strings are lowercase for robust case-insensitive substring matching.
AI_TOOL_KEYWORDS: List[str] = [
    "midjourney",
    "dall-e",
    "dalle",
    "stable diffusion",
    "firefly",
    "runway",
    "leonardo.ai",
    "playground ai",
    "ideogram",
    "flux",
    "chatgpt",
    "openai",
    "gpt-image",
    # Additional common variations and emerging tools
    "dall·e",
    "stablediffusion",
    "adobe firefly",
    "bing image creator",
    "craiyon",
    "comfyui",
    "automatic1111",
    "sora",
    "imagen",
]

# Canonical display mapping for cleaner explainability reporting
CANONICAL_TOOL_NAMES = {
    "midjourney": "Midjourney",
    "dall-e": "DALL-E",
    "dalle": "DALL-E",
    "dall·e": "DALL-E",
    "stable diffusion": "Stable Diffusion",
    "stablediffusion": "Stable Diffusion",
    "firefly": "Adobe Firefly",
    "adobe firefly": "Adobe Firefly",
    "runway": "Runway Gen",
    "leonardo.ai": "Leonardo.Ai",
    "playground ai": "Playground AI",
    "ideogram": "Ideogram",
    "flux": "FLUX",
    "chatgpt": "ChatGPT",
    "openai": "OpenAI",
    "gpt-image": "GPT-Image",
    "bing image creator": "Bing Image Creator",
    "craiyon": "Craiyon",
    "comfyui": "ComfyUI",
    "automatic1111": "AUTOMATIC1111 (Stable Diffusion WebUI)",
    "sora": "OpenAI Sora",
    "imagen": "Google Imagen",
}


def match_ai_keyword(text: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Checks whether the input text contains any known generative AI tool keyword.

    Args:
        text: Raw metadata string (e.g. EXIF Software tag, XMP CreatorTool, or C2PA claim generator).

    Returns:
        A tuple of (matched_raw_keyword, canonical_name) if a match is found,
        or None if no known AI keyword is present.
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip().lower()
    for kw in AI_TOOL_KEYWORDS:
        if kw in cleaned:
            canonical = CANONICAL_TOOL_NAMES.get(kw, kw.capitalize())
            return (kw, canonical)

    return None
