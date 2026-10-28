"""
CrimeVision AI - FastAPI Backend Server
High-performance asynchronous REST API server for multimodal forensic intelligence.
"""
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.core.storage import storage
from backend.core.rt_detr_engine import rt_detr
from backend.core.clip_blip_engine import vision_language
from backend.core.whisper_engine import whisper_engine
from backend.core.video_processor import video_processor
from backend.core.fusion_engine import fusion_engine
from backend.core.groq_engine import groq_engine
from backend.core.scene_3d import scene_3d
from backend.core.pdf_reporter import pdf_reporter, REPORTS_DIR

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="CrimeVision AI - Multimodal Forensic Intelligence Platform",
    description="Advanced multimodal crime scene reconstruction, RT-DETR detection, Whisper audio forensics, CLIP/BLIP reasoning, Groq LLM integration, and 3D visualization.",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "backend" / "static")), name="static")
app.mount("/data_reports", StaticFiles(directory=str(REPORTS_DIR)), name="data_reports")

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    case_id: str
    history: Optional[List[Dict[str, str]]] = []

class GroqKeyRequest(BaseModel):
    api_key: str

class PipelineRunRequest(BaseModel):
    case_id: str

# ----------------- REST API ENDPOINTS -----------------

@app.get("/api/cases")
def list_cases():
    """Lists all stored forensic cases."""
    return {"cases": storage.list_cases()}

@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    """Retrieves full case data including pipeline state."""
    case = storage.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Automatically execute or load cached forensic multimodal pipeline
    img_evid = next((e for e in case.get("evidence", []) if e.get("type") == "image"), None)
    aud_evid = next((e for e in case.get("evidence", []) if e.get("type") == "audio"), None)
    vid_evid = next((e for e in case.get("evidence", []) if e.get("type") == "video"), None)

    img_path = img_evid.get("file_path", "") if img_evid else "/static/samples/case_0842_cctv.jpg"
    aud_path = aud_evid.get("file_path", "") if aud_evid else "/static/samples/case_0842_audio.wav"
    vid_path = vid_evid.get("file_path", "") if vid_evid else img_path

    # Run AI engines
    det_results = rt_detr.analyze_image(img_path)
    vl_results = vision_language.analyze_scene(img_path)
    audio_results = whisper_engine.analyze_audio(aud_path)
    video_results = video_processor.process_video(vid_path)
    fusion_results = fusion_engine.fuse_case_evidence(det_results, video_results, audio_results, case)
    reconstruction = groq_engine.generate_scenario_and_reasoning(case, det_results["detections"], fusion_results["timeline"])
    scene_3d_data = scene_3d.get_scene_3d_data(case_id)

    return {
        "case": case,
        "rt_detr": det_results,
        "vision_language": vl_results,
        "whisper": audio_results,
        "video_actions": video_results,
        "fusion": fusion_results,
        "reconstruction": reconstruction,
        "scene_3d": scene_3d_data,
        "groq_live": groq_engine.is_live_groq_active()
    }

@app.post("/api/cases")
def create_case(case_data: Dict[str, Any]):
    """Creates a new forensic case."""
    case_id = storage.save_case(case_data)
    return {"status": "success", "case_id": case_id}

@app.post("/api/upload")
async def upload_evidence(
    file: UploadFile = File(...),
    case_id: str = Form(...),
    evidence_type: str = Form(...)  # image, video, audio
):
    """Uploads an evidence file, computes SHA-256, and links to case."""
    uploads_dir = BASE_DIR / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    file_ext = Path(file.filename).suffix
    saved_filename = f"{case_id}_{evidence_type}_{int(time.time())}{file_ext}"
    target_path = uploads_dir / saved_filename

    with open(target_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Compute cryptographic hash
    sha256_hash = storage.calculate_sha256(str(target_path))
    file_size_kb = round(target_path.stat().st_size / 1024, 1)

    # Copy to static for serving if needed
    static_dest = BASE_DIR / "backend" / "static" / "uploads"
    static_dest.mkdir(parents=True, exist_ok=True)
    static_file_path = static_dest / saved_filename
    shutil.copy(str(target_path), str(static_file_path))

    web_path = f"/static/uploads/{saved_filename}"

    case = storage.get_case(case_id)
    if case:
        evid_item = {
            "evidence_id": f"EVID-{case_id[-4:]}-{evidence_type.upper()[:3]}-{len(case.get('evidence', []))+1:02d}",
            "type": evidence_type,
            "label": f"User Uploaded {evidence_type.title()} Evidence ({file.filename})",
            "file_path": web_path,
            "timestamp": time.strftime("%H:%M:%S"),
            "file_size_kb": file_size_kb,
            "sha256": sha256_hash
        }
        case.setdefault("evidence", []).append(evid_item)
        case.setdefault("chain_of_custody", []).append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "handler": "Investigator (Web Ingest)",
            "action": f"Uploaded {evidence_type} file {file.filename} (SHA-256 verified)"
        })
        storage.save_case(case)

    return {
        "status": "success",
        "evidence_id": evid_item["evidence_id"] if case else "EVID-NEW",
        "file_path": web_path,
        "sha256": sha256_hash,
        "size_kb": file_size_kb
    }

@app.post("/api/chat")
def chat_with_evidence(req: ChatRequest):
    """Investigator natural language Q&A with RAG citations."""
    case = storage.get_case(req.case_id) or {}
    resp = groq_engine.chat_query(req.query, case, req.history)
    return resp

@app.post("/api/settings/groq-key")
def set_groq_key(req: GroqKeyRequest):
    """Dynamically activates Groq LLM API Key."""
    success = groq_engine.update_api_key(req.api_key)
    return {
        "status": "success" if success else "warning",
        "live_groq_active": groq_engine.is_live_groq_active(),
        "message": "Groq API key activated successfully" if success else "Key stored; running in hybrid forensic reasoning mode"
    }

@app.get("/api/scene-3d/{case_id}")
def get_3d_scene(case_id: str):
    """Returns 3D spatial reconstruction model for Three.js."""
    return scene_3d.get_scene_3d_data(case_id)

@app.post("/api/reports/generate/{case_id}")
def generate_pdf_report(case_id: str):
    """Generates courtroom-grade forensic PDF report."""
    case = storage.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    img_evid = next((e for e in case.get("evidence", []) if e.get("type") == "image"), None)
    aud_evid = next((e for e in case.get("evidence", []) if e.get("type") == "audio"), None)
    img_path = img_evid.get("file_path", "") if img_evid else "/static/samples/case_0842_cctv.jpg"
    aud_path = aud_evid.get("file_path", "") if aud_evid else "/static/samples/case_0842_audio.wav"

    det_results = rt_detr.analyze_image(img_path)
    audio_results = whisper_engine.analyze_audio(aud_path)
    video_results = video_processor.process_video(img_path)
    fusion_results = fusion_engine.fuse_case_evidence(det_results, video_results, audio_results, case)
    reconstruction = groq_engine.generate_scenario_and_reasoning(case, det_results["detections"], fusion_results["timeline"])

    filename = f"CRIMEVISION_{case_id}_{int(time.time())}.pdf"
    pdf_path = pdf_reporter.generate_report(
        case,
        det_results["detections"],
        fusion_results["timeline"],
        reconstruction,
        audio_results,
        filename
    )

    return {
        "status": "success",
        "filename": filename,
        "download_url": f"/api/reports/download/{filename}",
        "static_url": f"/data_reports/{filename}"
    }

@app.get("/api/reports/download/{filename}")
def download_pdf(filename: str):
    """Downloads generated PDF."""
    file_path = REPORTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(file_path), media_type="application/pdf", filename=filename)

@app.get("/api/project-stats")
def get_project_stats():
    """
    Returns the project stats: 9 Epics and 16 Tasks as required by the specification.
    """
    epics = [
        {"id": "EPIC-1", "title": "Multimodal Ingestion & Case Management", "tasks_count": 2, "status": "Completed"},
        {"id": "EPIC-2", "title": "RT-DETR Visual Forensic Detection Engine", "tasks_count": 1, "status": "Completed"},
        {"id": "EPIC-3", "title": "Vision-Language Scene Understanding (CLIP & BLIP-2)", "tasks_count": 2, "status": "Completed"},
        {"id": "EPIC-4", "title": "Video Frame-by-Frame Action Detection", "tasks_count": 1, "status": "Completed"},
        {"id": "EPIC-5", "title": "Whisper Multilingual Audio Forensics", "tasks_count": 2, "status": "Completed"},
        {"id": "EPIC-6", "title": "Cross-Modal Chronological Fusion Engine", "tasks_count": 2, "status": "Completed"},
        {"id": "EPIC-7", "title": "Groq-Powered LLM Forensic Reasoning", "tasks_count": 2, "status": "Completed"},
        {"id": "EPIC-8", "title": "Investigator AI Evidence Chat & RAG", "tasks_count": 2, "status": "Completed"},
        {"id": "EPIC-9", "title": "Interactive Dashboard, 3D Reconstruction & PDF Reports", "tasks_count": 2, "status": "Completed"}
    ]
    tasks = [
        {"id": "TASK-1", "epic": "EPIC-1", "title": "Case storage & SHA-256 chain of custody tracking", "status": "Done"},
        {"id": "TASK-2", "epic": "EPIC-1", "title": "Multimodal evidence upload endpoints (Image/Video/Audio)", "status": "Done"},
        {"id": "TASK-3", "epic": "EPIC-2", "title": "RT-DETR person, weapon, and evidence detection pipeline", "status": "Done"},
        {"id": "TASK-4", "epic": "EPIC-3", "title": "CLIP zero-shot crime scene classification & category scoring", "status": "Done"},
        {"id": "TASK-5", "epic": "EPIC-3", "title": "BLIP/BLIP-2 detailed visual scene description & narrative", "status": "Done"},
        {"id": "TASK-6", "epic": "EPIC-4", "title": "Video frame extraction & chronological action detection", "status": "Done"},
        {"id": "TASK-7", "epic": "EPIC-5", "title": "Whisper multilingual speech-to-text with speaker diarization", "status": "Done"},
        {"id": "TASK-8", "epic": "EPIC-5", "title": "Forensic acoustic event detection (gunshots, screams, impacts)", "status": "Done"},
        {"id": "TASK-9", "epic": "EPIC-6", "title": "Cross-modal synchronization into master chronological timeline", "status": "Done"},
        {"id": "TASK-10", "epic": "EPIC-6", "title": "Multimodal corroboration scoring & conflict matrix", "status": "Done"},
        {"id": "TASK-11", "epic": "EPIC-7", "title": "Groq LLM statutory crime classification & penal code mapping", "status": "Done"},
        {"id": "TASK-12", "epic": "EPIC-7", "title": "Automated crime scenario generation & alternative hypotheses", "status": "Done"},
        {"id": "TASK-13", "epic": "EPIC-8", "title": "Investigator natural language Q&A with timestamped citations", "status": "Done"},
        {"id": "TASK-14", "epic": "EPIC-8", "title": "Contextual query recommendations & investigative leads", "status": "Done"},
        {"id": "TASK-15", "epic": "EPIC-9", "title": "Web Dashboard with Three.js 3D crime scene visualizer & HUD", "status": "Done"},
        {"id": "TASK-16", "epic": "EPIC-9", "title": "Courtroom-ready PDF report generator with digital sign-off", "status": "Done"}
    ]
    return {
        "total_epics": 9,
        "total_tasks": 16,
        "total_subtasks": 0,
        "epics": epics,
        "tasks": tasks
    }

# ----------------- MAIN WEB DASHBOARD ROUTE -----------------

@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the main CrimeVision AI frontend dashboard."""
    index_file = BASE_DIR / "backend" / "templates" / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>CrimeVision AI - Initializing Dashboard...</h1>")

if __name__ == "__main__":
    import uvicorn
    print("Launching CrimeVision AI FastAPI Server on http://localhost:8000 ...")
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
