"""
HarvestLens AI - Application Launcher
Runs the FastAPI backend and serves the interactive frontend with loaded YOLOv8 model.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))


def check_prerequisites():
    """Verify presence of model weights and required directories."""
    model_primary = ROOT_DIR / "models" / "best.pt"
    model_fallback = ROOT_DIR / "best.pt"

    if not model_primary.exists() and not model_fallback.exists():
        print("\n[WARNING] 'best.pt' not found in either 'models/' or root folder.")
        print("Please place 'best.pt' in the project directory before running inference.\n")
    else:
        path = model_primary if model_primary.exists() else model_fallback
        print(f"[OK] YOLOv8 Model found at: {path}")

    # Ensure uploads folder
    (ROOT_DIR / "uploads" / "photos").mkdir(parents=True, exist_ok=True)
    print("[OK] Uploads directory verified.")


def main():
    print("\n" + "="*70)
    print(" HARVESTLENS AI - AGRICULTURAL INTELLIGENCE & DETECTION PLATFORM")
    print("="*70)
    check_prerequisites()

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))

    print(f"\n>> Serving Web Application at: http://{host}:{port}/")
    print(f">> Interactive Swagger API Docs: http://{host}:{port}/docs")
    print(f">> Direct HTML Entry Point: {ROOT_DIR / 'HarvestLens AI.html'}")
    print("\nPre-configured Demo Accounts:")
    print(" - Farmer:  farmer / farmer123password")
    print(" - Buyer:   buyer  / buyer123password")
    print(" - Admin:   admin  / admin123password\n" + "="*70 + "\n")

    uvicorn.run("backend.app.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
