import os
import json
import uuid
import base64
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import Batch, Observation, User, Listing
from backend.app.schemas import (
    BatchCreate,
    BatchUpdate,
    BatchResponse,
    ObservationCreate,
    ObservationResponse,
    DetectionItem
)
from backend.app.security import get_current_user, require_farmer
from backend.app.ml.detector import FruitDetector
from backend.app.config import UPLOADS_DIR

logger = logging.getLogger("harvestlens.batches")
router = APIRouter(prefix="/api/batches", tags=["Farmer Batches"])


def generate_batch_id() -> str:
    """Generate a clean human-readable batch ID like HL-A83B1."""
    return f"HL-{uuid.uuid4().hex[:5].upper()}"


def save_image_bytes(image_bytes: bytes, prefix: str = "img") -> str:
    """Save raw image bytes into the uploads directory with a unique filename."""
    sub_dir = UPLOADS_DIR / "photos"
    sub_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.jpg"
    filepath = sub_dir / filename
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return f"/uploads/photos/{filename}"


def serialize_observation(obs: Observation) -> ObservationResponse:
    """Helper to convert Observation model to response schema with parsed JSON detections."""
    try:
        dets = json.loads(obs.detections) if obs.detections else []
    except Exception:
        dets = []

    detection_items = [
        DetectionItem(
            label=d.get("label", ""),
            fruit=d.get("fruit", "Unknown"),
            maturity=d.get("maturity", "Unknown"),
            confidence=d.get("confidence", 0.0),
            box=d.get("box") or d.get("bbox") or [],
            class_id=d.get("class_id")
        ) for d in dets
    ]

    return ObservationResponse(
        id=obs.id,
        batch_id=obs.batch_id,
        image_path=obs.image_path,
        image_url=obs.image_path,
        detections=detection_items,
        dominant_maturity=obs.dominant_maturity,
        notes=obs.notes,
        analysed=obs.analysed,
        created_at=obs.created_at
    )


def serialize_batch(batch: Batch) -> BatchResponse:
    """Helper to convert Batch model to response schema with all observations."""
    obs_list = [serialize_observation(obs) for obs in batch.observations]

    listing_summary = None
    if batch.listing:
        listing_summary = {
            "id": batch.listing.id,
            "title": batch.listing.title,
            "status": batch.listing.status,
            "offered_qty": batch.listing.offered_qty,
            "price_per_unit": batch.listing.price_per_unit,
            "currency": batch.listing.currency,
            "region": batch.listing.region,
            "is_suspended": batch.listing.is_suspended,
            "suspension_reason": batch.listing.suspension_reason
        }

    return BatchResponse(
        id=batch.id,
        user_id=batch.user_id,
        name=batch.name,
        fruit=batch.fruit,
        variety=batch.variety,
        plot=batch.plot,
        notes=batch.notes,
        planted_date=batch.planted_date,
        fruit_set_date=batch.fruit_set_date,
        env_notes=batch.env_notes,
        interval_days=batch.interval_days,
        qty=batch.qty,
        unit=batch.unit,
        confirmed_category=batch.confirmed_category,
        is_archived=batch.is_archived,
        is_completed=batch.is_completed,
        created_at=batch.created_at,
        updated_at=batch.updated_at,
        observations=obs_list,
        listing=listing_summary
    )


@router.get("", response_model=List[BatchResponse])
def get_farmer_batches(
    include_archived: bool = Query(False),
    fruit: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List private batches. Farmers see ONLY their own private batches.
    Buyers cannot access this endpoint. Admins can view for moderation purposes.
    """
    if current_user.role == "buyer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Buyers do not have access to private farmer batch management."
        )

    query = db.query(Batch)
    if current_user.role != "admin":
        query = query.filter(Batch.user_id == current_user.id)

    if not include_archived:
        query = query.filter(Batch.is_archived == False)

    if fruit and fruit != "All":
        query = query.filter(Batch.fruit == fruit)

    batches = query.order_by(Batch.created_at.desc()).all()
    return [serialize_batch(b) for b in batches]


@router.post("", response_model=BatchResponse, status_code=status.HTTP_201_CREATED)
def create_batch(
    req: BatchCreate,
    current_user: User = Depends(require_farmer),
    db: Session = Depends(get_db)
):
    """Create a new private farmer batch with optional initial observation."""
    batch_id = generate_batch_id()

    # Ensure batch ID is unique
    while db.query(Batch).filter(Batch.id == batch_id).first():
        batch_id = generate_batch_id()

    new_batch = Batch(
        id=batch_id,
        user_id=current_user.id,
        name=req.name.strip(),
        fruit=req.fruit,
        variety=req.variety,
        plot=req.plot,
        notes=req.notes,
        planted_date=req.planted_date,
        fruit_set_date=req.fruit_set_date,
        env_notes=req.env_notes,
        interval_days=req.interval_days,
        qty=req.qty,
        unit=req.unit,
        confirmed_category=req.confirmed_category,
        is_archived=False,
        is_completed=False
    )
    db.add(new_batch)
    db.flush()

    # Handle initial observation if provided
    if req.initial_observation:
        obs_req = req.initial_observation
        image_path = None

        if obs_req.image_base64:
            try:
                # Remove header if present (e.g. data:image/jpeg;base64,...)
                raw_data = obs_req.image_base64
                if "," in raw_data:
                    raw_data = raw_data.split(",", 1)[1]
                img_bytes = base64.b64decode(raw_data)
                image_path = save_image_bytes(img_bytes, prefix=f"batch_{batch_id}")
            except Exception as e:
                logger.warning(f"Could not save base64 image: {e}")

        dets_raw = [d.model_dump() if hasattr(d, 'model_dump') else d.dict() for d in (obs_req.detections or [])]
        dominant = FruitDetector.calculate_dominant_maturity(dets_raw)

        obs = Observation(
            batch_id=batch_id,
            image_path=image_path,
            detections=json.dumps(dets_raw),
            dominant_maturity=dominant,
            notes=obs_req.notes or "Initial assessment",
            analysed=obs_req.analysed
        )
        db.add(obs)

    db.commit()
    db.refresh(new_batch)
    return serialize_batch(new_batch)


@router.get("/{batch_id}", response_model=BatchResponse)
def get_batch_details(
    batch_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve single batch and its full observation timeline.
    Strictly verifies ownership: Farmer A cannot view Farmer B's private batch.
    """
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' not found."
        )

    # Server-side privacy check
    if current_user.role != "admin" and batch.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have permission to view this private batch."
        )

    return serialize_batch(batch)


@router.put("/{batch_id}", response_model=BatchResponse)
def update_batch(
    batch_id: str,
    req: BatchUpdate,
    current_user: User = Depends(require_farmer),
    db: Session = Depends(get_db)
):
    """Update private batch details (inventory, confirmed maturity, status)."""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' not found."
        )

    if current_user.role != "admin" and batch.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You cannot modify another farmer's batch."
        )

    update_data = req.model_dump(exclude_unset=True) if hasattr(req, 'model_dump') else req.dict(exclude_unset=True)
    for field, val in update_data.items():
        if val is not None:
            setattr(batch, field, val)

    batch.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(batch)
    return serialize_batch(batch)


@router.post("/{batch_id}/observations", response_model=ObservationResponse)
async def add_observation(
    batch_id: str,
    file: Optional[UploadFile] = File(None),
    notes: Optional[str] = Form(None),
    analysed: bool = Form(True),
    current_user: User = Depends(require_farmer),
    db: Session = Depends(get_db)
):
    """Add a new observation photograph with automatic YOLOv8 maturity detection."""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' not found."
        )

    if current_user.role != "admin" and batch.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You cannot add observations to another farmer's batch."
        )

    image_path = None
    detections: List[dict] = []
    dominant = None

    if file:
        content = await file.read()
        image_path = save_image_bytes(content, prefix=f"batch_{batch_id}")

        if analysed:
            try:
                detector = FruitDetector.get_instance()
                pred = detector.predict(content)
                detections = pred.get("detections", [])
                dominant = FruitDetector.calculate_dominant_maturity(detections)
            except Exception as e:
                logger.error(f"Inference during observation recording failed: {e}")

    obs = Observation(
        batch_id=batch_id,
        image_path=image_path,
        detections=json.dumps(detections),
        dominant_maturity=dominant,
        notes=notes,
        analysed=analysed
    )
    db.add(obs)
    batch.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(obs)

    return serialize_observation(obs)


@router.delete("/{batch_id}")
def archive_batch(
    batch_id: str,
    current_user: User = Depends(require_farmer),
    db: Session = Depends(get_db)
):
    """Archive a batch so it is hidden from the main view."""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found")

    if current_user.role != "admin" and batch.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    batch.is_archived = True
    db.commit()
    return {"message": f"Batch '{batch_id}' has been archived."}
