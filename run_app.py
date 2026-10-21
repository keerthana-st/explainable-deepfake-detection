"""
Launcher script to run both FastAPI backend and Vite frontend together.
"""
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

def main():
    print("=" * 60)
    print("Starting Explainable Deepfake Detection Platform")
    print("=" * 60)

    # 1. Start Backend
    print("[1/2] Starting FastAPI backend on http://localhost:8000 ...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(PROJECT_ROOT)
    )

    time.sleep(2)

    # 2. Start Frontend
    print("[2/2] Starting Vite React frontend on http://localhost:5173 ...")
    frontend_proc = subprocess.Popen(
        ["npm.cmd" if sys.platform == "win32" else "npm", "run", "dev"],
        cwd=str(FRONTEND_DIR)
    )

    print("\n" + "=" * 60)
    print("Platform running! Open your browser at: http://localhost:5173")
    print("Press Ctrl+C to stop both servers.")
    print("=" * 60 + "\n")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
