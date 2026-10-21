import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.config import BASE_DIR, UPLOADS_DIR
from backend.app.database import engine, Base, SessionLocal
from backend.app.ml.detector import FruitDetector
from backend.app.routers.auth import get_or_create_demo_user
from backend.app.models import Batch, Observation, Listing, User
import json

from backend.app.routers import (
    auth,
    predict,
    batches,
    marketplace,
    requests,
    messages,
    admin
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("harvestlens.main")


def seed_initial_demo_data():
    """Seed initial demo users and a sample batch if database is empty."""
    db = SessionLocal()
    try:
        farmer = get_or_create_demo_user(db, "farmer")
        get_or_create_demo_user(db, "buyer")
        get_or_create_demo_user(db, "admin")

        # Check if farmer has any batches
        existing_batch = db.query(Batch).filter(Batch.user_id == farmer.id).first()
        if not existing_batch:
            logger.info("Seeding initial sample batch for demo farmer...")
            sample_batch = Batch(
                id="HL-DEMO1",
                user_id=farmer.id,
                name="Mango Batch 001",
                fruit="Mango",
                variety="Alphonso",
                plot="North Orchard - Block B",
                notes="Early season blossoming, healthy irrigation cycle.",
                planted_date="2024-03-15",
                fruit_set_date="2026-02-10",
                env_notes="Warm dry week, 28-34°C with gentle breezes.",
                interval_days=5,
                qty=1200.0,
                unit="kg",
                confirmed_category="Ripe",
                is_archived=False,
                is_completed=False
            )
            db.add(sample_batch)
            db.flush()

            # Seed initial observation
            dets = [
                {"label": "Mango Ripe", "fruit": "Mango", "maturity": "Ripe", "confidence": 0.92, "box": [80, 60, 280, 270]},
                {"label": "Mango Unripe", "fruit": "Mango", "maturity": "Unripe", "confidence": 0.85, "box": [310, 100, 480, 290]}
            ]
            obs = Observation(
                batch_id=sample_batch.id,
                image_path=None,
                detections=json.dumps(dets),
                dominant_maturity="Ripe",
                notes="Initial orchard maturity assessment: high bloom, excellent color transition.",
                analysed=True
            )
            db.add(obs)

            # Publish sample batch to marketplace
            listing = Listing(
                batch_id=sample_batch.id,
                farmer_id=farmer.id,
                title="Premium Alphonso Mangoes - Fresh Harvest",
                fruit="Mango",
                confirmed_category="Ripe",
                variety="Alphonso",
                region="Ratnagiri / Konkan Belt",
                offered_qty=800.0,
                price_per_unit=180.0,
                currency="INR",
                available_from="2026-10-25",
                farm_name=farmer.farm_name,
                contact_phone=farmer.phone,
                contact_email=farmer.email,
                description="Naturally tree-ripened Alphonso mangoes ready for fresh retail or premium pulp. Sweet aroma with vibrant golden-orange blush.",
                status="published"
            )
            db.add(listing)
            db.commit()
            logger.info("Demo data seeded successfully.")
    except Exception as e:
        logger.error(f"Error seeding demo data: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events:
    1. Initialize database schema
    2. Load YOLOv8 model ONCE into memory at backend startup
    3. Seed demo accounts and sample batch
    """
    logger.info("Starting HarvestLens AI application...")
    # 1. Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified.")

    # 2. Load model once
    try:
        detector = FruitDetector.get_instance()
        logger.info(f"YOLOv8 FruitDetector initialized with {len(detector.names)} classes.")
    except Exception as e:
        logger.error(f"Could not load YOLOv8 model at startup: {e}")

    # 3. Seed demo accounts
    seed_initial_demo_data()

    yield

    logger.info("HarvestLens AI application shutdown.")


app = FastAPI(
    title="HarvestLens AI - Agricultural Intelligence & Ripeness Detection Platform",
    description="Full-stack AI platform for fruit maturity classification, private batch tracking, and agricultural exchange.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware - allows frontend from file://, localhost:3000, 5500, 8000, etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Uploads directory for static photo serving
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Include Routers
app.include_router(predict.router)
app.include_router(auth.router)
app.include_router(batches.router)
app.include_router(marketplace.router)
app.include_router(requests.router)
app.include_router(messages.router)
app.include_router(admin.router)

# Mount Frontend directory if exists
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def serve_index():
    """Serve the main frontend application."""
    frontend_index = FRONTEND_DIR / "index.html"
    root_html = BASE_DIR / "HarvestLens AI.html"

    if frontend_index.exists():
        return FileResponse(frontend_index)
    elif root_html.exists():
        return FileResponse(root_html)
    return {"message": "HarvestLens AI Backend is running. Please access /docs for API documentation."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
