"""
Global vs. Regional Consistency Analyzer for the Secondary System.
Evaluates whether localized regional evidence supports, partially supports,
or contradicts the Primary System global verdict.
Strictly enforces the Non-Override Policy: Never mutates the global verdict.
"""

from typing import List, Tuple
from .models import ConsistencyStatus, RegionEvidence, RegionalConsistencyResult, RegionVerdict


class RegionalConsistencyAnalyzer:
    """
    Forensic consistency evaluator comparing global decision against localized evidence.
    """

    def analyze_consistency(
        self,
        primary_global_verdict: str,
        region_evidences: List[RegionEvidence],
    ) -> RegionalConsistencyResult:
        """
        Evaluate global vs regional forensic concordance.

        Args:
            primary_global_verdict: Primary verdict string (e.g. 'FAKE', 'REAL', 'AI-EDITED', 'UNCERTAIN')
            region_evidences: List of evaluated RegionEvidence objects

        Returns:
            RegionalConsistencyResult with status, explanation, and audit flags
        """
        clean_global = primary_global_verdict.upper().strip()

        # Categorize regions
        suspicious = [r.region_id for r in region_evidences if r.region_verdict == RegionVerdict.AI_EDITED.value]
        authentic = [r.region_id for r in region_evidences if r.region_verdict == RegionVerdict.REAL.value]
        uncertain = [r.region_id for r in region_evidences if r.region_verdict in [RegionVerdict.UNCERTAIN.value, RegionVerdict.INSUFFICIENT_SIZE.value]]

        is_global_fake = any(k in clean_global for k in ["FAKE", "AI-EDITED", "AI-GENERATED", "MANIPULATED", "SYNTHETIC"])
        is_global_real = any(k in clean_global for k in ["REAL", "AUTHENTIC", "NATURAL"])
        is_global_uncertain = "UNCERTAIN" in clean_global or "SUSPICIOUS" in clean_global

        status = ConsistencyStatus.INCONCLUSIVE
        requires_review = False
        warning = None
        explanation = ""

        # Case 1: Global Verdict is FAKE / AI-EDITED
        if is_global_fake:
            if len(suspicious) > 0:
                status = ConsistencyStatus.SUPPORTED
                susp_names = ", ".join(suspicious)
                auth_names = ", ".join(authentic) if authentic else "None"
                explanation = (
                    f"SUPPORTED: The global {primary_global_verdict} verdict is strongly supported by localized "
                    f"evidence. Suspicious artifacts were isolated in: [{susp_names}]. "
                    f"Authentic regions: [{auth_names}]."
                )
            elif len(uncertain) > 0 and len(authentic) == 0:
                status = ConsistencyStatus.PARTIALLY_SUPPORTED
                explanation = (
                    f"PARTIALLY SUPPORTED: Global verdict indicated manipulation, but regional analysis "
                    f"yielded borderline/uncertain evidence across segmented regions."
                )
            else:
                status = ConsistencyStatus.CONTRADICTED
                requires_review = True
                warning = "REGIONAL INCONSISTENCY: Global verdict is FAKE, but all segmented regions test as REAL."
                explanation = (
                    f"CONTRADICTED: Global verdict is {primary_global_verdict}, but all segmented regions "
                    f"[{', '.join(authentic)}] exhibit authentic photographic characteristics. Flagged for review."
                )

        # Case 2: Global Verdict is REAL / AUTHENTIC
        elif is_global_real:
            if len(suspicious) == 0:
                status = ConsistencyStatus.SUPPORTED
                explanation = (
                    f"SUPPORTED: Global REAL verdict is corroborated by regional analysis. "
                    f"All analyzed regions [{', '.join(authentic)}] show consistent natural optical features."
                )
            else:
                status = ConsistencyStatus.CONTRADICTED
                requires_review = True
                warning = "REGIONAL INCONSISTENCY: Global verdict is REAL, but suspicious anomalies were isolated in specific regions."
                explanation = (
                    f"CONTRADICTED: Global verdict is {primary_global_verdict}, but regional analysis flagged "
                    f"synthetic manipulation in: [{', '.join(suspicious)}]. Potential localized inpainting."
                )

        # Case 3: Global Verdict is UNCERTAIN
        else:
            if len(suspicious) > 0 and len(authentic) > 0:
                status = ConsistencyStatus.PARTIALLY_SUPPORTED
                explanation = (
                    f"EXPLAINABLE RESOLUTION: Regional analysis explains global uncertainty. "
                    f"Suspicious region(s) [{', '.join(suspicious)}] coexist with authentic background/regions "
                    f"[{', '.join(authentic)}], confirming localized regional manipulation."
                )
            elif len(suspicious) > 0:
                status = ConsistencyStatus.PARTIALLY_SUPPORTED
                explanation = (
                    f"PARTIALLY SUPPORTED: Global verdict was uncertain, but regional evidence highlights "
                    f"elevated synthesis artifacts in [{', '.join(suspicious)}]."
                )
            else:
                status = ConsistencyStatus.INCONCLUSIVE
                explanation = (
                    "INCONCLUSIVE: Both global and regional evidence fall within borderline margin. "
                    "Manual forensic inspection recommended."
                )

        return RegionalConsistencyResult(
            status=status,
            primary_verdict=primary_global_verdict,
            explanation=explanation,
            suspicious_regions=suspicious,
            authentic_regions=authentic,
            requires_review=requires_review,
            inconsistency_warning=warning,
        )

    # Convenience alias
    analyze = analyze_consistency

