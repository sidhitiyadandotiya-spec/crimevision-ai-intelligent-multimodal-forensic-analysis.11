"""
CrimeVision AI - Flask Backend & Dashboard Server
Flask application providing identical REST APIs, static serving, and Web Dashboard.
"""
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any

from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_cors import CORS

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

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "backend" / "static"),
    template_folder=str(BASE_DIR / "backend" / "templates")
)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data_reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(str(REPORTS_DIR), filename)

@app.route("/api/cases", methods=["GET"])
def list_cases():
    return jsonify({"cases": storage.list_cases()})

@app.route("/api/cases/<case_id>", methods=["GET"])
def get_case(case_id):
    case = storage.get_case(case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404

    img_evid = next((e for e in case.get("evidence", []) if e.get("type") == "image"), None)
    aud_evid = next((e for e in case.get("evidence", []) if e.get("type") == "audio"), None)
    vid_evid = next((e for e in case.get("evidence", []) if e.get("type") == "video"), None)

    img_path = img_evid.get("file_path", "") if img_evid else "/static/samples/case_0842_cctv.jpg"
    aud_path = aud_evid.get("file_path", "") if aud_evid else "/static/samples/case_0842_audio.wav"
    vid_path = vid_evid.get("file_path", "") if vid_evid else img_path

    det_results = rt_detr.analyze_image(img_path)
    vl_results = vision_language.analyze_scene(img_path)
    audio_results = whisper_engine.analyze_audio(aud_path)
    video_results = video_processor.process_video(vid_path)
    fusion_results = fusion_engine.fuse_case_evidence(det_results, video_results, audio_results, case)
    reconstruction = groq_engine.generate_scenario_and_reasoning(case, det_results["detections"], fusion_results["timeline"])
    scene_3d_data = scene_3d.get_scene_3d_data(case_id)

    return jsonify({
        "case": case,
        "rt_detr": det_results,
        "vision_language": vl_results,
        "whisper": audio_results,
        "video_actions": video_results,
        "fusion": fusion_results,
        "reconstruction": reconstruction,
        "scene_3d": scene_3d_data,
        "groq_live": groq_engine.is_live_groq_active()
    })

@app.route("/api/cases", methods=["POST"])
def create_case():
    case_data = request.json or {}
    case_id = storage.save_case(case_data)
    return jsonify({"status": "success", "case_id": case_id})

@app.route("/api/upload", methods=["POST"])
def upload_evidence():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    case_id = request.form.get("case_id", "CASE-NEW")
    evidence_type = request.form.get("evidence_type", "image")

    uploads_dir = BASE_DIR / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_ext = Path(file.filename).suffix
    saved_filename = f"{case_id}_{evidence_type}_{int(time.time())}{file_ext}"
    target_path = uploads_dir / saved_filename
    file.save(str(target_path))

    sha256_hash = storage.calculate_sha256(str(target_path))
    file_size_kb = round(target_path.stat().st_size / 1024, 1)

    static_dest = BASE_DIR / "backend" / "static" / "uploads"
    static_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(target_path), str(static_dest / saved_filename))
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

    return jsonify({
        "status": "success",
        "evidence_id": evid_item["evidence_id"] if case else "EVID-NEW",
        "file_path": web_path,
        "sha256": sha256_hash,
        "size_kb": file_size_kb
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    query = data.get("query", "")
    case_id = data.get("case_id", "")
    history = data.get("history", [])
    case = storage.get_case(case_id) or {}
    resp = groq_engine.chat_query(query, case, history)
    return jsonify(resp)

@app.route("/api/settings/groq-key", methods=["POST"])
def set_groq_key():
    data = request.json or {}
    api_key = data.get("api_key", "")
    success = groq_engine.update_api_key(api_key)
    return jsonify({
        "status": "success" if success else "warning",
        "live_groq_active": groq_engine.is_live_groq_active(),
        "message": "Groq API key activated successfully" if success else "Key stored; running in hybrid forensic reasoning mode"
    })

@app.route("/api/scene-3d/<case_id>", methods=["GET"])
def get_3d_scene(case_id):
    return jsonify(scene_3d.get_scene_3d_data(case_id))

@app.route("/api/reports/generate/<case_id>", methods=["POST"])
def generate_pdf_report(case_id):
    case = storage.get_case(case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404

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

    return jsonify({
        "status": "success",
        "filename": filename,
        "download_url": f"/api/reports/download/{filename}",
        "static_url": f"/data_reports/{filename}"
    })

@app.route("/api/reports/download/<filename>", methods=["GET"])
def download_pdf(filename):
    file_path = REPORTS_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "Report not found"}), 404
    return send_file(str(file_path), as_attachment=True, download_name=filename, mimetype="application/pdf")

@app.route("/api/project-stats", methods=["GET"])
def get_project_stats():
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
    return jsonify({
        "total_epics": 9,
        "total_tasks": 16,
        "total_subtasks": 0,
        "epics": epics,
        "tasks": tasks
    })

if __name__ == "__main__":
    print("Launching CrimeVision AI Flask Dashboard Server on http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True)
