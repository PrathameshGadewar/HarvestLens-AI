import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

# Model path resolution: check models/best.pt first, then root best.pt
MODELS_DIR = BASE_DIR / "models"
PRIMARY_MODEL_PATH = MODELS_DIR / "best.pt"
FALLBACK_MODEL_PATH = BASE_DIR / "best.pt"

if PRIMARY_MODEL_PATH.exists():
    MODEL_PATH = PRIMARY_MODEL_PATH
elif FALLBACK_MODEL_PATH.exists():
    MODEL_PATH = FALLBACK_MODEL_PATH
else:
    MODEL_PATH = PRIMARY_MODEL_PATH

# Storage
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'harvestlens.db'}")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "harvestlens-ai-secret-key-production-farm-auth-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Inference Defaults
DEFAULT_CONFIDENCE = float(os.getenv("DEFAULT_CONFIDENCE", "0.25"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_SIZE_MB = 15

# Supported Fruits & Maturity Stages
FRUITS = ["Apple", "Banana", "Grape", "Mango", "Melon", "Orange", "Peach", "Pear"]
MATURITY_STAGES = ["Unripe", "Ripe", "Overripe", "Rotten"]
