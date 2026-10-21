"""
FastAPI Backend for Multi-Agent Explainable Deepfake Detection Platform.
Provides REST APIs for:
  - Image forensic analysis (Multi-agent pipeline)
  - Analysis history & summary statistics
  - System health and architecture status
  - Static visualization file serving
"""

import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy scalars and arrays to native Python primitives for clean JSON encoding."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from integration.complete_pipeline import CompleteDeepfakeDetectionPipeline
from secondary_system.models import SecondaryTriggerMode

app = FastAPI(
    title="Explainable Deepfake Detection API",
    description="Multi-Agent Forensic Deepfake Detection Platform API",
    version="2.0.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output directories exist
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
HISTORY_FILE = OUTPUTS_DIR / "history.json"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Mount outputs directory to serve visualizations
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Initialize Pipeline once
pipeline: Optional[CompleteDeepfakeDetectionPipeline] = None


def get_pipeline() -> CompleteDeepfakeDetectionPipeline:
    global pipeline
    if pipeline is None:
        pipeline = CompleteDeepfakeDetectionPipeline(
            secondary_trigger_mode=SecondaryTriggerMode.ALWAYS,
            classifier_weight=0.60,
            frequency_weight=0.40,
            enable_secondary=True,
            enable_gradcam=True,
        )
    return pipeline


# Benchmark Presets available on local machine
PRESETS = {
    "portrait_real": {
        "id": "portrait_real",
        "name": "Authentic Portrait (img2.jpeg)",
        "path": r"C:\Users\KEERTHANA S.T\OneDrive\Pictures\Saved Pictures\img2.jpeg",
        "description": "High-resolution real human portrait captured with camera optics",
    },
    "animal_ai": {
        "id": "animal_ai",
        "name": "AI-Generated Dog (dog.jpg)",
        "path": r"D:\dog.jpg",
        "description": "Photorealistic AI-generated animal image with diffusion artifacts",
    },
    "benchmark_cifake": {
        "id": "benchmark_cifake",
        "name": "CIFAKE Authentic Benchmark (0000 (6).jpg)",
        "path": r"D:\archive\test\REAL\0000 (6).jpg",
        "description": "Kaggle CIFAKE test authentic sample image",
    },
    "wallpaper_synthetic": {
        "id": "wallpaper_synthetic",
        "name": "Upsampled Wallpaper (walpa2.jpeg)",
        "path": str(PROJECT_ROOT / "walpa2.jpeg"),
        "description": "Synthetically upsampled scene with periodic grid artifacts",
    },
}


def load_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history_entry(entry: Dict[str, Any]):
    history = load_history()
    # Insert newest at beginning
    history.insert(0, entry)
    # Keep last 100 entries
    history = history[:100]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)


def format_visualization_urls(vis_dict: Dict[str, Any], bust_cache: bool = True) -> Dict[str, str]:
    """Convert local filesystem paths to web-accessible URLs with optional cache-busting."""
    web_vis = {}
    ts = int(time.time() * 1000)
    for key, val in vis_dict.items():
        if val and isinstance(val, str):
            p = Path(val)
            url = None
            if p.exists():
                try:
                    rel = p.relative_to(OUTPUTS_DIR)
                    url = f"/outputs/{rel.as_posix()}"
                except ValueError:
                    # If not under outputs dir, check filename directly in outputs
                    if (OUTPUTS_DIR / p.name).exists():
                        url = f"/outputs/{p.name}"
                    else:
                        url = f"/outputs/{p.name}"
            else:
                url = f"/outputs/{p.name}"

            if url:
                if bust_cache:
                    url = f"{url}?t={ts}"
                web_vis[key] = url
    return web_vis


@app.get("/api/system-status")
def get_system_status():
    return {
        "status": "online",
        "primary_system": "Online",
        "secondary_system": "Online",
        "ai_model": "Ready",
        "device": "AMD Ryzen 5 CPU (Multi-threaded)",
        "classifier": "EfficientNet-B0 (Two-Stage Fine-Tuned, Calibrated)",
        "frequency_agent": "2D-DCT 28-D Power Spectral Anomaly",
        "metadata_agent": "4-Tier C2PA / EXIF / DQT Hardware Matcher",
        "disagreement_layer": "Zero-Fusion Categorical Arbitration Active",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/stats")
def get_stats():
    history = load_history()
    total = len(history)
    if total == 0:
        return {
            "total_analyzed": 0,
            "ai_generated_count": 0,
            "ai_generated_pct": 0.0,
            "ai_edited_count": 0,
            "ai_edited_pct": 0.0,
            "real_count": 0,
            "real_pct": 0.0,
            "uncertain_count": 0,
            "uncertain_pct": 0.0,
            "recent_week_count": 0,
        }

    ai_gen = sum(1 for h in history if "AI-GENERATED" in h.get("global_verdict", "").upper() or "FAKE" in h.get("global_verdict", "").upper())
    ai_edit = sum(1 for h in history if "AI-EDITED" in h.get("global_verdict", "").upper())
    real = sum(1 for h in history if "REAL" in h.get("global_verdict", "").upper() or "AUTHENTIC" in h.get("global_verdict", "").upper())
    unc = sum(1 for h in history if "UNCERTAIN" in h.get("global_verdict", "").upper() or "SUSPICIOUS" in h.get("global_verdict", "").upper())

    return {
        "total_analyzed": total,
        "ai_generated_count": ai_gen,
        "ai_generated_pct": round((ai_gen / total) * 100, 1),
        "ai_edited_count": ai_edit,
        "ai_edited_pct": round((ai_edit / total) * 100, 1),
        "real_count": real,
        "real_pct": round((real / total) * 100, 1),
        "uncertain_count": unc,
        "uncertain_pct": round((unc / total) * 100, 1),
        "recent_week_count": total,
    }


@app.get("/api/history")
def get_history():
    history = load_history()
    summaries = []
    for h in history:
        summaries.append({
            "id": h.get("id"),
            "timestamp": h.get("timestamp"),
            "filename": h.get("filename"),
            "global_verdict": h.get("global_verdict"),
            "confidence": h.get("confidence"),
            "confidence_display": h.get("confidence_display"),
            "region_count": h.get("region_count", 0),
            "suspicious_region_count": h.get("suspicious_region_count", 0),
            "consistency_status": h.get("consistency_status", "N/A"),
            "preview_url": h.get("preview_url"),
        })
    return summaries


@app.get("/api/history/{entry_id}")
def get_history_detail(entry_id: str):
    history = load_history()
    for h in history:
        if h.get("id") == entry_id:
            return h
    raise HTTPException(status_code=404, detail="Analysis record not found")


@app.get("/api/presets")
def get_presets():
    available = []
    for k, v in PRESETS.items():
        exists = os.path.exists(v["path"])
        available.append({
            "id": v["id"],
            "name": v["name"],
            "description": v["description"],
            "available": exists,
        })
    return available


class PresetRequest(BaseModel):
    preset_id: str


PRESET_ALIASES = {
    "img2": "portrait_real",
    "dog": "animal_ai",
    "cifake": "benchmark_cifake",
    "walpa2": "wallpaper_synthetic",
}


@app.post("/api/analyze-preset")
def analyze_preset(req: PresetRequest):
    pid = PRESET_ALIASES.get(req.preset_id.lower().strip(), req.preset_id)
    if pid not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Invalid preset ID '{req.preset_id}'. Available: {list(PRESETS.keys())}")
    preset_info = PRESETS[pid]
    image_path = preset_info["path"]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Preset file not found on disk: {image_path}")

    return execute_analysis(image_path, filename=Path(image_path).name)


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Validate extension
    ext = Path(file.filename or "image.png").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG images are supported.")

    # Save to uploads dir with unique filename
    unique_id = str(uuid.uuid4())[:8]
    safe_name = f"{Path(file.filename).stem}_{unique_id}{ext}"
    dest_path = UPLOADS_DIR / safe_name

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return execute_analysis(str(dest_path), filename=file.filename or safe_name)


def execute_analysis(image_path: str, filename: str) -> Dict[str, Any]:
    analysis_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    p = get_pipeline()
    res = p.analyze(image_path, output_dir=OUTPUTS_DIR, analysis_id=analysis_id)
    elapsed = time.perf_counter() - t0

    if res.get("status") == "error":
        raise HTTPException(status_code=422, detail=res.get("error_message", "Analysis failed"))

    timestamp = datetime.now().isoformat()

    primary = res.get("primary_system", {})
    sec = res.get("secondary_system", {})

    # Build web URLs for visualizations
    raw_vis = res.get("visualizations", {})
    web_vis = format_visualization_urls(raw_vis)

    # Build web URLs for preprocessing representations
    raw_prep = res.get("preprocessing", {})
    if raw_prep and "saved_paths" in raw_prep:
        web_prep = format_visualization_urls(raw_prep["saved_paths"])
        raw_prep["original_url"] = web_prep.get("original")
        raw_prep["rgb_url"] = web_prep.get("rgb")
        raw_prep["grayscale_url"] = web_prep.get("grayscale")
        raw_prep["resized_url"] = web_prep.get("resized")

    # Build web URLs for regional crops and heatmaps
    if sec and "regions" in sec:
        for r in sec["regions"]:
            if isinstance(r, dict):
                if r.get("crop_path"):
                    c_vis = format_visualization_urls({"crop": r["crop_path"]})
                    r["crop_url"] = c_vis.get("crop")
                if r.get("heatmap_path"):
                    h_vis = format_visualization_urls({"heatmap": r["heatmap_path"]})
                    r["heatmap_url"] = h_vis.get("heatmap")

    # Determine preview URL
    preview_url = raw_prep.get("original_url") or web_vis.get("region_overlay") or web_vis.get("gradcam_global")

    consistency_status = "N/A"
    region_count = 0
    susp_count = 0
    if sec and sec.get("executed"):
        region_count = sec.get("region_count", 0)
        susp_count = sec.get("suspicious_region_count", 0)
        if sec.get("regional_consistency"):
            consistency_status = sec["regional_consistency"].get("status", "N/A")

    response_payload = {
        "id": analysis_id,
        "timestamp": timestamp,
        "filename": filename,
        "latency_seconds": round(elapsed, 2),
        "status": "success",
        "global_verdict": primary.get("global_verdict"),
        "confidence": primary.get("confidence"),
        "confidence_display": primary.get("confidence_display"),
        "decision_basis": primary.get("decision_basis"),
        "badge": primary.get("badge"),
        "region_count": region_count,
        "suspicious_region_count": susp_count,
        "consistency_status": consistency_status,
        "preview_url": preview_url,
        "preprocessing": raw_prep,
        "primary_system": primary,
        "secondary_system": sec,
        "visualizations": web_vis,
        "final_explanation": res.get("final_explanation", ""),
    }

    # Sanitize NumPy types for clean JSON encoding
    response_payload = sanitize_for_json(response_payload)

    # Save to history
    save_history_entry(response_payload)

    return response_payload


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
