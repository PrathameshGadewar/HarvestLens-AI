import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_

from backend.app.database import get_db
from backend.app.models import Batch, Listing, PurchaseRequest, User
from backend.app.schemas import ListingCreate, ListingUpdate, ListingResponse
from backend.app.security import get_current_user, require_farmer, get_optional_user

logger = logging.getLogger("harvestlens.marketplace")
router = APIRouter(prefix="/api/marketplace", tags=["Agricultural Marketplace"])


def calculate_reserved_qty(listing_id: int, db: Session) -> float:
    """Calculate the total quantity reserved by accepted purchase requests."""
    accepted_requests = db.query(PurchaseRequest).filter(
        PurchaseRequest.listing_id == listing_id,
        PurchaseRequest.status == "Accepted"
    ).all()
    return sum(req.quantity for req in accepted_requests)


def serialize_listing(listing: Listing, db: Session) -> ListingResponse:
    """Convert Listing model to schema with calculated reserved and available quantity."""
    reserved = calculate_reserved_qty(listing.id, db)
    available = max(0.0, listing.offered_qty - reserved)

    # Latest observation photo
    latest_img = None
    if listing.batch and listing.batch.observations:
        for obs in reversed(listing.batch.observations):
            if obs.image_path:
                latest_img = obs.image_path
                break

    return ListingResponse(
        id=listing.id,
        batch_id=listing.batch_id,
        farmer_id=listing.farmer_id,
        title=listing.title,
        fruit=listing.fruit,
        confirmed_category=listing.confirmed_category,
        variety=listing.variety,
        region=listing.region,
        offered_qty=listing.offered_qty,
        reserved_qty=round(reserved, 2),
        available_qty=round(available, 2),
        unit=listing.batch.unit if listing.batch else "kg",
        price_per_unit=listing.price_per_unit,
        currency=listing.currency,
        available_from=listing.available_from,
        farm_name=listing.farm_name or (listing.farmer.farm_name if listing.farmer else None),
        contact_phone=listing.contact_phone,
        contact_email=listing.contact_email,
        description=listing.description,
        status=listing.status,
        is_suspended=listing.is_suspended,
        suspension_reason=listing.suspension_reason,
        published_at=listing.published_at,
        created_at=listing.created_at,
        latest_image_url=latest_img
    )


@router.get("/listings", response_model=List[ListingResponse])
def get_public_listings(
    q: Optional[str] = Query(None, description="Search by title, variety or region"),
    fruit: Optional[str] = Query(None, description="Filter by fruit variety"),
    maturity: Optional[str] = Query(None, description="Filter by confirmed maturity category"),
    region: Optional[str] = Query(None, description="Filter by geographic region"),
    max_price: Optional[float] = Query(None, description="Maximum price per unit"),
    db: Session = Depends(get_db)
):
    """
    Public marketplace feed. Only explicitly published, active, non-suspended listings appear.
    Private farmer batches and draft listings never appear here.
    """
    query = db.query(Listing).join(Batch).filter(
        Listing.status == "published",
        Listing.is_suspended == False,
        Batch.is_archived == False
    )

    if q:
        search = f"%{q.strip()}%"
        query = query.filter(
            or_(
                Listing.title.ilike(search),
                Listing.variety.ilike(search),
                Listing.region.ilike(search),
                Listing.fruit.ilike(search)
            )
        )

    if fruit and fruit != "All":
        query = query.filter(Listing.fruit == fruit)

    if maturity and maturity != "All":
        query = query.filter(Listing.confirmed_category == maturity)

    if region and region.strip():
        query = query.filter(Listing.region.ilike(f"%{region.strip()}%"))

    if max_price is not None:
        query = query.filter(Listing.price_per_unit <= max_price)

    listings = query.order_by(Listing.published_at.desc(), Listing.created_at.desc()).all()
    return [serialize_listing(l, db) for l in listings]


@router.get("/listings/{listing_id}", response_model=ListingResponse)
def get_listing_detail(listing_id: int, db: Session = Depends(get_db)):
    """Retrieve full details of an active public listing."""
    listing = db.query(Listing).filter(Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found.")

    if listing.status != "published" or listing.is_suspended:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This listing is currently unavailable or has been archived."
        )

    return serialize_listing(listing, db)


@router.post("/listings/{batch_id}/publish", response_model=ListingResponse)
def publish_batch_listing(
    batch_id: str,
    req: ListingCreate,
    current_user: User = Depends(require_farmer),
    db: Session = Depends(get_db)
):
    """
    Farmer explicitly publishes a private batch to the HarvestLens Exchange.
    Validates inventory, confirmed maturity category, and batch ownership.
    """
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found.")

    if current_user.role != "admin" and batch.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You can only publish listings for your own batches."
        )

    if not batch.qty or batch.qty <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch quantity must be greater than 0 before publishing to the marketplace."
        )

    if not batch.confirmed_category:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please set and confirm the batch maturity category (Unripe, Ripe, Overripe, Rotten) before publishing."
        )

    if req.offered_qty > batch.qty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Offered quantity ({req.offered_qty} {batch.unit}) cannot exceed total batch quantity ({batch.qty} {batch.unit})."
        )

    # Check if a listing already exists for this batch
    listing = db.query(Listing).filter(Listing.batch_id == batch_id).first()
    new_status = "published" if req.publish_now else "draft"

    if listing:
        if listing.is_suspended:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"This listing has been suspended by an administrator: {listing.suspension_reason}"
            )
        listing.title = req.title.strip()
        listing.fruit = batch.fruit
        listing.confirmed_category = batch.confirmed_category
        listing.variety = batch.variety
        listing.region = req.region.strip()
        listing.offered_qty = req.offered_qty
        listing.price_per_unit = req.price_per_unit
        listing.currency = req.currency
        listing.available_from = req.available_from
        listing.farm_name = req.farm_name or current_user.farm_name
        listing.contact_phone = req.contact_phone or current_user.phone
        listing.contact_email = req.contact_email or current_user.email
        listing.description = req.description
        listing.status = new_status
        if new_status == "published" and not listing.published_at:
            listing.published_at = datetime.utcnow()
        listing.updated_at = datetime.utcnow()
    else:
        listing = Listing(
            batch_id=batch.id,
            farmer_id=current_user.id,
            title=req.title.strip(),
            fruit=batch.fruit,
            confirmed_category=batch.confirmed_category,
            variety=batch.variety,
            region=req.region.strip(),
            offered_qty=req.offered_qty,
            price_per_unit=req.price_per_unit,
            currency=req.currency,
            available_from=req.available_from,
            farm_name=req.farm_name or current_user.farm_name,
            contact_phone=req.contact_phone or current_user.phone,
            contact_email=req.contact_email or current_user.email,
            description=req.description,
            status=new_status,
            published_at=datetime.utcnow() if new_status == "published" else None
        )
        db.add(listing)

    db.commit()
    db.refresh(listing)
    return serialize_listing(listing, db)


@router.put("/listings/{listing_id}/status", response_model=ListingResponse)
def update_listing_status(
    listing_id: int,
    status_update: ListingUpdate,
    current_user: User = Depends(require_farmer),
    db: Session = Depends(get_db)
):
    """Update listing status (pause, resume, mark sold out, or change details)."""
    listing = db.query(Listing).filter(Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found.")

    if current_user.role != "admin" and listing.farmer_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    if listing.is_suspended:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot update suspended listing. Contact support or admin."
        )

    if status_update.status:
        listing.status = status_update.status
        if status_update.status == "published" and not listing.published_at:
            listing.published_at = datetime.utcnow()

    update_fields = status_update.model_dump(exclude_unset=True, exclude={"status"}) if hasattr(status_update, 'model_dump') else status_update.dict(exclude_unset=True, exclude={"status"})
    for k, v in update_fields.items():
        if v is not None:
            setattr(listing, k, v)

    listing.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(listing)
    return serialize_listing(listing, db)
