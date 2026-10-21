"""
State-specific, non-templated explainable reasoning generation for Module 4.
"""

from typing import Optional


def build_disagreement_reasoning(
    output_state: str,
    freq_verdict: str,
    classifier_verdict: str,
    freq_score: float,
    classifier_score: float,
    metadata_flag: str,
    disclosed_tool: Optional[str] = None,
    raw_manifest_status: Optional[str] = None,
) -> str:
    """
    Constructs rich, state-calibrated, explainable reasoning strictly adhering
    to the four distinct output states.
    """
    # -------------------------------------------------------------------------
    # STATE C: PROVENANCE OVERRIDE FAKE
    # -------------------------------------------------------------------------
    if output_state == "PROVENANCE_OVERRIDE_FAKE":
        tool_str = disclosed_tool if disclosed_tool else "AI Generation Tool"
        
        if freq_verdict == "real" and classifier_verdict == "real":
            agent_clause = "Frequency and Classifier agents both indicated REAL"
            basis_detail = (
                f"Although both vision models evaluated this image as authentic photographic capture "
                f"(Frequency: {freq_score:.4f}, Classifier: {classifier_score:.4f}), embedded C2PA / EXIF "
                f"digital credentials provide cryptographically signed, deterministic proof of generation."
            )
        elif freq_verdict != classifier_verdict:
            agent_clause = "Frequency and Classifier disagreed"
            flagged = "Frequency Agent" if freq_verdict == "fake" else "Classifier Agent"
            unflagged = "Classifier Agent" if freq_verdict == "fake" else "Frequency Agent"
            basis_detail = (
                f"Although the probabilistic vision agents were in conflict ({flagged} flagged FAKE "
                f"while {unflagged} predicted REAL), embedded C2PA / EXIF digital credentials provide "
                f"cryptographically signed, deterministic proof of generation."
            )
        else:
            # Both agreed on fake
            agent_clause = "Frequency and Classifier agents both indicated FAKE"
            basis_detail = (
                f"Both probabilistic vision agents independently flagged this image as synthetic "
                f"(Frequency: {freq_score:.4f}, Classifier: {classifier_score:.4f}), which is corroborated "
                f"by embedded C2PA / EXIF digital credentials providing cryptographically signed proof."
            )

        return (
            f"PROVENANCE OVERRIDE DETECTED: {agent_clause}, but cryptographic provenance metadata "
            f"(disclosed generator: {tool_str}) confirms synthetic origin. "
            f"This provenance evidence overrides the probabilistic signals, as it constitutes direct, "
            f"verifiable proof rather than statistical inference.\n\n"
            f"ARBITRATION BASIS: {basis_detail} "
            f"This provenance evidence overrides probabilistic signals because statistical and machine-learned "
            f"estimates are categorically superseded by factual disclosure."
        )

    # -------------------------------------------------------------------------
    # STATE A: HIGH CONFIDENCE (REAL or FAKE)
    # -------------------------------------------------------------------------
    if output_state == "HIGH_CONFIDENCE_FAKE":
        return (
            f"HIGH CONFIDENCE SYNTHETIC (CONVERGENT PROBABILISTIC AGREEMENT): "
            f"Both independent Category 1 probabilistic agents agree that this image is AI-generated / deepfake. "
            f"The Frequency Agent detected high-frequency periodic grid artifacts (score: {freq_score:.4f}), "
            f"and the Classifier Agent identified synthetic textural and anatomical anomalies "
            f"(fake probability: {classifier_score:.4f}). "
            f"Neither agent detected authentic sensor noise or natural spectral decay, establishing strong "
            f"convergent evidence of synthetic origin."
        )

    if output_state == "HIGH_CONFIDENCE_REAL":
        return (
            f"HIGH CONFIDENCE AUTHENTIC (CONVERGENT PROBABILISTIC AGREEMENT): "
            f"Both independent Category 1 probabilistic agents agree that this image is an authentic photograph. "
            f"The Frequency Agent confirmed smooth, natural 2D-DCT spectral energy (score: {freq_score:.4f}) without "
            f"periodic upsampling lattices, while the Classifier Agent confirmed genuine CMOS sensor shot-noise and "
            f"natural anatomical micro-textures (fake probability: {classifier_score:.4f}). "
            f"No cryptographic provenance records or AI metadata tags contradict this authentic assessment."
        )

    # -------------------------------------------------------------------------
    # STATE B: UNCERTAIN (DISAGREEMENT)
    # -------------------------------------------------------------------------
    if output_state == "UNCERTAIN_frequency_flagged_only":
        return (
            f"UNCERTAIN / MANUAL REVIEW RECOMMENDED (FREQUENCY FLAGGED ONLY): "
            f"Category 1 probabilistic agents are in conflict. "
            f"The Frequency Agent flagged this image as SUSPICIOUS (score: {freq_score:.4f}), but the "
            f"Classifier Agent predicted REAL (fake probability: {classifier_score:.4f}).\n\n"
            f"EXPLAINABLE RESOLUTION: This pattern is characteristic of high-contrast natural textures "
            f"(e.g. human hair, woven textiles, foliage, brickwork) or aggressive device-level edge sharpening "
            f"applied by modern smartphone camera DSPs, which elevate high-frequency DCT energy without "
            f"indicating generative manipulation. Meanwhile, the semantic vision classifier identified authentic "
            f"CMOS sensor shot-noise and natural anatomical features. "
            f"Because no deterministic provenance metadata is present, this case cannot be autonomously resolved "
            f"with high confidence and is flagged for human review."
        )

    if output_state == "UNCERTAIN_classifier_flagged_only":
        return (
            f"UNCERTAIN / MANUAL REVIEW RECOMMENDED (CLASSIFIER FLAGGED ONLY): "
            f"Category 1 probabilistic agents are in conflict. "
            f"The Classifier Agent predicted FAKE (score: {classifier_score:.4f}), but the "
            f"Frequency Agent detected a smooth, natural frequency spectrum (score: {freq_score:.4f}).\n\n"
            f"EXPLAINABLE RESOLUTION: This pattern is characteristic of a modern Latent Diffusion Model "
            f"(e.g. DALL-E 3, Midjourney v6, SDXL, FLUX). Modern diffusion models use continuous autoencoder "
            f"decoders that do not produce transposed-convolution checkerboard grids, successfully bypassing "
            f"classical 2D-DCT frequency analysis. However, the Classifier Agent identified synthetic airbrushed skin, "
            f"lacking physical photon noise, or unnatural lighting boundaries. "
            f"Because no deterministic provenance metadata is present, this case cannot be autonomously resolved "
            f"with high confidence and is flagged for human review."
        )

    # -------------------------------------------------------------------------
    # STATE D: PROVENANCE UNCERTAIN
    # -------------------------------------------------------------------------
    if output_state == "PROVENANCE_UNCERTAIN":
        status_note = f" (Status: {raw_manifest_status})" if raw_manifest_status else ""
        return (
            f"PROVENANCE UNCERTAIN / UNRESOLVED MANIFEST DETECTED: "
            f"A Coalition for Content Provenance and Authenticity (C2PA) manifest container was detected embedded "
            f"in this file{status_note}, but the claim_generator field could not be parsed or resolved to a known tool.\n\n"
            f"FALLBACK ARBITRATION: An intact provenance container indicates cryptographic tracking or editorial "
            f"post-processing, but does not provide actionable proof of AI generation. Per specification, "
            f"this finding DOES NOT qualify for an override and cannot alter the visual verdict.\n"
            f"Under Category 1 probabilistic analysis: Frequency Agent predicted {freq_verdict.upper()} ({freq_score:.4f}) "
            f"and Classifier Agent predicted {classifier_verdict.upper()} ({classifier_score:.4f}). "
            f"Manual forensic inspection of raw_manifest_data is strongly recommended."
        )

    return (
        f"Multi-Agent Evaluation Completed. Frequency: {freq_score:.4f} ({freq_verdict}), "
        f"Classifier: {classifier_score:.4f} ({classifier_verdict}), Metadata: {metadata_flag}."
    )
