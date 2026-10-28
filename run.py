"""
CrimeVision AI - Universal Server Launcher
Allows running either FastAPI (Default: Port 8000) or Flask (Port 5000).

Usage:
    py run.py          # Launches FastAPI server on http://localhost:8000
    py run.py --flask  # Launches Flask server on http://localhost:5000
"""
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="CrimeVision AI Platform Launcher")
    parser.add_argument("--flask", action="store_true", help="Launch Flask server instead of FastAPI")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default 8000 for FastAPI, 5000 for Flask)")
    args = parser.parse_args()

    if args.flask:
        port = args.port or 5000
        print(f"\n========================================================")
        print(f"  CRIMEVISION AI - FLASK FORENSIC DASHBOARD SERVER")
        print(f"  Endpoint: http://localhost:{port}")
        print(f"  REST APIs & 3D WebGL Visualization Active")
        print(f"========================================================\n")
        from backend.flask_app import app
        app.run(host=args.host, port=port, debug=False)
    else:
        port = args.port or 8000
        print(f"\n========================================================")
        print(f"  CRIMEVISION AI - FASTAPI MULTIMODAL INTELLIGENCE SERVER")
        print(f"  Web Dashboard: http://localhost:{port}")
        print(f"  Interactive OpenAPI Docs: http://localhost:{port}/docs")
        print(f"  RT-DETR + Whisper + CLIP/BLIP-2 + Groq LLM + Three.js")
        print(f"========================================================\n")
        import uvicorn
        uvicorn.run("backend.fastapi_app:app", host=args.host, port=port, reload=False)

if __name__ == "__main__":
    main()
