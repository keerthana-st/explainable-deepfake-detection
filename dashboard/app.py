"""
Streamlit Forensic Dashboard for Explainable Deepfake Detection.
Multi-Agent Architecture: Preprocessing -> Primary System -> Secondary System -> Unified Decision.
"""

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
import streamlit as st


# Configure page early
st.set_page_config(
    page_title="Multi-Agent Deepfake Detection Forensic Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .verdict-box-real {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        padding: 1.25rem 1.75rem;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
        margin-bottom: 1rem;
    }
    .verdict-box-fake {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        padding: 1.25rem 1.75rem;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.25);
        margin-bottom: 1rem;
    }
    .verdict-box-uncertain {
        background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
        color: white;
        padding: 1.25rem 1.75rem;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.25);
        margin-bottom: 1rem;
    }
    .agent-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .consistency-badge-supported {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .consistency-badge-contradicted {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .consistency-badge-partial {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



@st.cache_resource(show_spinner="Initializing Multi-Agent Detection Pipeline...")
def load_pipeline(
    classifier_weight: float = 0.60,
    frequency_weight: float = 0.40,
    enable_gradcam: bool = True,
):
    from integration.complete_pipeline import CompleteDeepfakeDetectionPipeline
    from secondary_system.models import SecondaryTriggerMode

    return CompleteDeepfakeDetectionPipeline(
        secondary_trigger_mode=SecondaryTriggerMode.ALWAYS,
        classifier_weight=classifier_weight,
        frequency_weight=frequency_weight,
        enable_secondary=True,
        enable_gradcam=enable_gradcam,
    )


def generate_html_report(result: Dict[str, Any]) -> str:
    """Generates an executive forensic audit HTML report."""
    primary = result.get("primary_system", {})
    sec = result.get("secondary_system", {})
    regions = sec.get("regions", []) if sec else []
    consistency = sec.get("regional_consistency", {}) if sec else {}

    rows = ""
    for r in regions:
        badge_color = "#10B981" if r["verdict"] == "REAL" else ("#EF4444" if r["verdict"] == "AI-EDITED" else "#F59E0B")
        rows += f"""
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><b>{r['region_id']}</b></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{r['region_type']}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{r['classifier_score']*100:.1f}%</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{r['frequency_score']*100:.1f}%</td>
            <td style="padding: 8px; border: 1px solid #ddd;"><span style="color: {badge_color}; font-weight: bold;">{r['verdict']}</span></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{r['agreement']}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Forensic Deepfake Detection Audit Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; margin: 30px; }}
            h1 {{ color: #1E293B; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; }}
            h2 {{ color: #334155; margin-top: 24px; }}
            .card {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin: 16px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
            th {{ background: #F1F5F9; text-align: left; padding: 10px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>🛡️ Multi-Agent Deepfake Detection Forensic Audit Report</h1>
        <p><b>Target File:</b> {result.get('file', 'Target Image')} | <b>Generated At:</b> System Runtime</p>
        
        <div class="card">
            <h2>1. Global Forensic Finding (WHAT)</h2>
            <p><b>Global Verdict:</b> <span style="font-size: 1.2em; font-weight: bold;">{primary.get('global_verdict')}</span></p>
            <p><b>Calibrated Confidence:</b> {primary.get('confidence_display')}</p>
            <p><b>Arbitration Basis:</b> {primary.get('decision_basis')} ({primary.get('badge')})</p>
        </div>

        <div class="card">
            <h2>2. Primary Multi-Agent Findings (WHY)</h2>
            <ul>
                <li><b>Vision Classifier (EfficientNet-B0):</b> {primary.get('classifier', {}).get('prediction')} (Score: {primary.get('classifier', {}).get('score', 0)*100:.1f}%, Disparity Ratio: {primary.get('classifier', {}).get('disparity_ratio', 1.0)})</li>
                <li><b>Frequency Agent (DCT Analysis):</b> Anomaly Score: {primary.get('frequency', {}).get('score', 0)*100:.1f}% (Suspicious: {primary.get('frequency', {}).get('is_suspicious')})</li>
                <li><b>Metadata & Provenance Agent:</b> State: {primary.get('metadata', {}).get('flag')} (Disclosed Tool: {primary.get('metadata', {}).get('disclosed_tool') or 'None'}, C2PA: {primary.get('metadata', {}).get('c2pa_present')})</li>
            </ul>
        </div>

        <div class="card">
            <h2>3. Regional Analysis Breakdown (WHERE)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Region ID</th>
                        <th>Type</th>
                        <th>Classifier Fake %</th>
                        <th>Frequency Score %</th>
                        <th>Regional Verdict</th>
                        <th>Agent Agreement</th>
                    </tr>
                </thead>
                <tbody>
                    {rows if rows else '<tr><td colspan="6" style="padding: 10px; text-align: center;">No sub-regions analyzed</td></tr>'}
                </tbody>
            </table>
            <p style="margin-top: 12px;"><b>Global-Regional Consistency:</b> {consistency.get('status', 'N/A')} - {consistency.get('explanation', '')}</p>
        </div>

        <div class="card">
            <h2>4. Forensic Narrative Dossier</h2>
            <pre style="white-space: pre-wrap; font-family: inherit;">{result.get('final_explanation', 'No narrative generated.')}</pre>
        </div>
    </body>
    </html>
    """
    return html


def main():
    st.markdown('<div class="main-header">🛡️ Explainable Deepfake Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-Agent Forensic Architecture: Frequency Analysis + EfficientNet-B0 Classification + Metadata Provenance + Region-Level Localization</div>',
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # SIDEBAR CONFIGURATION
    # -------------------------------------------------------------------------
    st.sidebar.title("⚙️ System Configuration")

    st.sidebar.subheader("Multi-Agent Architecture")
    st.sidebar.markdown(
        """
        - **Module 0**: Forensic Preprocessing
        - **Agent 1**: Frequency Domain (2D-DCT)
        - **Agent 2**: Deep Vision (EfficientNet-B0)
        - **Agent 3**: Metadata & C2PA Provenance
        - **Module 4**: Disagreement Layer
        - **Secondary**: Regional Localization
        """
    )

    st.sidebar.subheader("Inference Settings")
    trigger_mode_str = st.sidebar.selectbox(
        "Secondary Trigger Mode",
        options=["ALWAYS", "LOW_CONFIDENCE", "DISAGREEMENT", "SUSPICIOUS_ONLY"],
        index=0,
        help="Controls when the Secondary Region Analysis pipeline is triggered.",
    )
    from secondary_system.models import SecondaryTriggerMode
    mode_map = {
        "ALWAYS": SecondaryTriggerMode.ALWAYS,
        "LOW_CONFIDENCE": SecondaryTriggerMode.LOW_CONFIDENCE,
        "DISAGREEMENT": SecondaryTriggerMode.DISAGREEMENT,
        "SUSPICIOUS_ONLY": SecondaryTriggerMode.SUSPICIOUS_ONLY,
    }
    trigger_mode = mode_map[trigger_mode_str]

    cls_weight = st.sidebar.slider("Classifier Weight (w_cls)", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    freq_weight = st.sidebar.slider("Frequency Weight (w_freq)", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
    enable_gradcam = st.sidebar.checkbox("Compute Grad-CAM Attribution", value=True)

    # Load cached pipeline
    pipeline = load_pipeline(
        classifier_weight=cls_weight,
        frequency_weight=freq_weight,
        enable_gradcam=enable_gradcam,
    )
    pipeline.secondary_pipeline.trigger_mode = trigger_mode

    # -------------------------------------------------------------------------
    # SECTION 1: IMAGE INGESTION & PREVIEW
    # -------------------------------------------------------------------------
    st.markdown("### 1. Image Ingestion & Input Selection")

    input_tab1, input_tab2 = st.tabs(["⚡ Quick Test Presets", "📁 Upload Custom Image"])

    selected_image_path: Optional[str] = None
    image_display: Optional[Image.Image] = None

    preset_options = {
        "User Authentic Portrait (img2.jpeg)": r"C:\Users\KEERTHANA S.T\OneDrive\Pictures\Saved Pictures\img2.jpeg",
        "AI-Generated Animal (dog.jpg)": r"D:\dog.jpg",
        "CIFAKE Benchmark Authentic (0000 (6).jpg)": r"D:\archive\test\REAL\0000 (6).jpg",
        "Workspace Sample (walpa2.jpeg)": "walpa2.jpeg",
    }

    with input_tab1:
        st.write("Select a pre-verified test image from your local environment:")
        selected_preset_name = st.selectbox("Choose Benchmark Preset", list(preset_options.keys()))
        candidate_path = preset_options[selected_preset_name]
        if os.path.exists(candidate_path):
            selected_image_path = candidate_path
        else:
            st.warning(f"File not found on machine: {candidate_path}")

    with input_tab2:
        uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Save uploaded bytes to outputs temp dir to preserve EXIF / C2PA
            temp_dir = Path("outputs/uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file_path = temp_dir / uploaded_file.name
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_image_path = str(temp_file_path)

    # Show preview and image metadata
    if selected_image_path and os.path.exists(selected_image_path):
        col_prev1, col_prev2 = st.columns([1, 2])
        try:
            image_display = Image.open(selected_image_path)
            with col_prev1:
                st.image(image_display, caption=f"Selected: {Path(selected_image_path).name}", use_container_width=True)
            with col_prev2:
                st.write("**File Metadata Summary:**")
                st.write(f"- **Path:** `{selected_image_path}`")
                st.write(f"- **Dimensions:** {image_display.width} × {image_display.height} px")
                st.write(f"- **Format:** {image_display.format or 'Unknown'}")
                st.write(f"- **File Size:** {os.path.getsize(selected_image_path) / 1024:.2f} KB")
        except Exception as e:
            st.error(f"Error loading image preview: {e}")

    analyze_clicked = st.button("🚀 Run Deepfake Forensic Analysis", type="primary", use_container_width=True)

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    if analyze_clicked and selected_image_path:
        with st.spinner("Analyzing image across Frequency, Vision, Metadata, and Regional domains..."):
            res = pipeline.analyze(selected_image_path, output_dir="outputs")
            st.session_state["last_result"] = res

    result = st.session_state.get("last_result")

    if result:
        st.divider()
        primary = result["primary_system"]
        secondary = result.get("secondary_system")
        visualizations = result.get("visualizations", {})

        # ---------------------------------------------------------------------
        # SECTION 2: PRIMARY SYSTEM GLOBAL RESULT
        # ---------------------------------------------------------------------
        st.markdown("### 2. Primary System Global Result")
        verdict = primary["global_verdict"]
        conf_display = primary["confidence_display"]

        if "REAL" in verdict:
            box_cls = "verdict-box-real"
            icon = "✅"
        elif "FAKE" in verdict or "AI-GENERATED" in verdict:
            box_cls = "verdict-box-fake"
            icon = "🚨"
        else:
            box_cls = "verdict-box-uncertain"
            icon = "⚠️"

        st.markdown(
            f"""
            <div class="{box_cls}">
                <div style="font-size: 1.6rem; font-weight: 800;">{icon} {verdict}</div>
                <div style="font-size: 1.1rem; opacity: 0.95; margin-top: 4px;">
                    Calibrated Confidence: <b>{conf_display}</b> | Arbitration Basis: <b>{primary['decision_basis']}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.markdown(
                f"""
                <div class="agent-card">
                    <h4>📸 Vision Classifier</h4>
                    <p><b>Prediction:</b> {primary['classifier']['prediction']}</p>
                    <p><b>Fake Probability:</b> {primary['classifier']['score']*100:.1f}%</p>
                    <p><b>Noise Variance:</b> {primary['classifier']['noise_variance']}</p>
                    <p><b>Disparity Ratio:</b> {primary['classifier']['disparity_ratio']}×</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_a2:
            st.markdown(
                f"""
                <div class="agent-card">
                    <h4>〰️ Frequency Agent</h4>
                    <p><b>Anomaly Score:</b> {primary['frequency']['score']*100:.1f}%</p>
                    <p><b>Suspicious Flag:</b> {primary['frequency']['is_suspicious']}</p>
                    <p><b>Raw Energy:</b> {primary['frequency']['raw_energy']}</p>
                    <p><b>Status:</b> {'Grid Artifacts' if primary['frequency']['is_suspicious'] else 'Natural Decay'}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_a3:
            st.markdown(
                f"""
                <div class="agent-card">
                    <h4>🔍 Metadata & C2PA</h4>
                    <p><b>Provenance Flag:</b> {primary['metadata']['flag']}</p>
                    <p><b>Disclosed Tool:</b> {primary['metadata']['disclosed_tool'] or 'None'}</p>
                    <p><b>C2PA Signed:</b> {primary['metadata']['c2pa_present']}</p>
                    <p><b>DQT Hardware:</b> {primary['metadata']['dqt_profile'] or 'Standard/Stripped'}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------------------
        # SECTION 3: REGIONAL ANALYSIS TABLE
        # ---------------------------------------------------------------------
        st.markdown("### 3. Regional Analysis Breakdown")
        if secondary and secondary.get("executed") and secondary.get("regions"):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Regions", secondary.get("region_count", 0))
            m2.metric("Suspicious Regions", secondary.get("suspicious_region_count", 0))
            clean_count = sum(1 for r in secondary["regions"] if r["verdict"] == "REAL")
            m3.metric("Authentic Regions", clean_count)
            uncertain_count = sum(1 for r in secondary["regions"] if r["verdict"] == "UNCERTAIN")
            m4.metric("Uncertain Regions", uncertain_count)

            table_data = []
            for r in secondary["regions"]:
                table_data.append({
                    "Region ID": r["region_id"],
                    "Region Type": r["region_type"],
                    "Classifier Fake %": f"{r['classifier_score']*100:.1f}%",
                    "Frequency Score %": f"{r['frequency_score']*100:.1f}%",
                    "Combined Suspicion %": f"{r['combined_score']*100:.1f}%",
                    "Agent Agreement": r["agreement"],
                    "Regional Verdict": r["verdict"],
                    "Confidence": f"{r['confidence']*100:.1f}%",
                })
            st.dataframe(table_data, use_container_width=True)
        else:
            st.info("Secondary region analysis was not triggered under current trigger criteria.")

        # ---------------------------------------------------------------------
        # SECTION 4: VISUAL REGION OVERLAY
        # ---------------------------------------------------------------------
        st.markdown("### 4. Visual Region Overlay (Side-by-Side)")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.write("**Preprocessed Full Input Image:**")
            if image_display:
                st.image(image_display, use_container_width=True)
        with col_v2:
            st.write("**Color-Coded Regional Boundaries & Verdicts:**")
            overlay_path = visualizations.get("region_overlay")
            if overlay_path and os.path.exists(overlay_path):
                st.image(overlay_path, use_container_width=True)
            else:
                st.info("Region overlay visualization not generated.")

        # ---------------------------------------------------------------------
        # SECTION 5: SPATIAL HEATMAP & GRAD-CAM ATTRIBUTION
        # ---------------------------------------------------------------------
        st.markdown("### 5. Spatial Suspicion Heatmap & Model Attribution")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.write("**Continuous Spatial Suspicion Density Heatmap:**")
            heat_path = visualizations.get("spatial_heatmap")
            if heat_path and os.path.exists(heat_path):
                st.image(heat_path, use_container_width=True)
            else:
                st.info("Spatial heatmap not available.")
        with col_h2:
            st.write("**Selective Regional Grad-CAM Visual Attribution:**")
            grad_path = visualizations.get("gradcam_regional") or visualizations.get("gradcam_global")
            if grad_path and os.path.exists(grad_path):
                st.image(grad_path, use_container_width=True)
            else:
                st.info("Grad-CAM visualization not generated.")

        # Spectral analysis row
        with st.expander("📊 View Frequency Spectral Plots (2D-DCT & Radial Energy Decay)"):
            col_sp1, col_sp2 = st.columns(2)
            spec_path = visualizations.get("dct_spectrum")
            rad_path = visualizations.get("radial_decay")
            if spec_path and os.path.exists(spec_path):
                with col_sp1:
                    st.image(spec_path, caption="2D Discrete Cosine Transform (DCT) Log Spectrum", use_container_width=True)
            if rad_path and os.path.exists(rad_path):
                with col_sp2:
                    st.image(rad_path, caption="Radial Frequency Energy Decay Profile", use_container_width=True)

        # ---------------------------------------------------------------------
        # SECTION 6: METADATA & PROVENANCE AUDIT
        # ---------------------------------------------------------------------
        st.markdown("### 6. Metadata & Hardware Provenance Audit")
        meta = primary["metadata"]
        with st.container():
            st.markdown(
                f"""
                - **Provenance Status:** `{meta['flag']}`
                - **Disclosed Generative Software:** `{meta['disclosed_tool'] or 'None Detected'}`
                - **C2PA Cryptographic Signature:** `{'Manifest Verified' if meta['c2pa_present'] else 'No Manifest Found'}`
                - **JPEG Quantization Table Match:** `{meta['dqt_profile'] or 'Camera Default / Standard'}`
                - **Forensic Reasoning:** {meta['reasoning']}
                """
            )

        # ---------------------------------------------------------------------
        # SECTION 7: GLOBAL-REGIONAL CONSISTENCY ANALYSIS
        # ---------------------------------------------------------------------
        st.markdown("### 7. Global–Regional Consistency Analysis")
        if secondary and secondary.get("regional_consistency"):
            c_info = secondary["regional_consistency"]
            c_status = c_info["status"]
            if c_status == "SUPPORTED":
                c_badge = '<span class="consistency-badge-supported">CONSISTENCY: SUPPORTED</span>'
            elif c_status == "CONTRADICTED":
                c_badge = '<span class="consistency-badge-contradicted">CONSISTENCY: CONTRADICTED (AUDIT ALERT)</span>'
            else:
                c_badge = '<span class="consistency-badge-partial">CONSISTENCY: PARTIAL / INCONCLUSIVE</span>'

            st.markdown(f"Status: {c_badge}", unsafe_allow_html=True)
            st.write(f"**Explanation:** {c_info.get('explanation')}")
            if c_info.get("warning"):
                st.warning(c_info["warning"])
            st.caption("🔒 Non-Override Invariant Enforced: Regional consistency audits support explainability without mutating the global verdict.")
        else:
            st.info("Regional consistency evaluation was not executed.")

        # ---------------------------------------------------------------------
        # SECTION 8: FORENSIC EVIDENCE NARRATIVE (WHAT, WHERE, WHY, HOW)
        # ---------------------------------------------------------------------
        st.markdown("### 8. Forensic Evidence Dossier (WHAT, WHERE, WHY, HOW)")
        st.text_area("Multi-Agent Explainability Narrative", value=result.get("final_explanation", ""), height=180)

        # ---------------------------------------------------------------------
        # SECTION 9: DOWNLOADABLE FORENSIC REPORT
        # ---------------------------------------------------------------------
        st.markdown("### 9. Downloadable Forensic Reports")
        col_d1, col_d2 = st.columns(2)

        # JSON Export
        json_str = json.dumps(result, indent=2, default=str)
        with col_d1:
            st.download_button(
                label="📥 Download JSON Forensic Audit Trail",
                data=json_str,
                file_name="forensic_deepfake_report.json",
                mime="application/json",
                use_container_width=True,
            )

        # HTML Export
        html_report = generate_html_report(result)
        with col_d2:
            st.download_button(
                label="📄 Download Printable HTML Audit Report",
                data=html_report,
                file_name="forensic_deepfake_report.html",
                mime="text/html",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
