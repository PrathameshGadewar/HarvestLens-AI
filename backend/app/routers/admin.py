import logging
from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import Listing, AuditLog, User, Batch, Observation, PurchaseRequest
from backend.app.schemas import (
    ListingResponse,
    ModerationAction,
    AuditLogResponse
)
from backend.app.security import require_admin
from backend.app.routers.marketplace import serialize_listing

logger = logging.getLogger("harvestlens.admin")
router = APIRouter(prefix="/api/admin", tags=["Admin Moderation & Governance"])


@router.get("/listings", response_model=List[ListingResponse])
def get_all_marketplace_listings(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin view: inspect all marketplace listings across all statuses and moderation states."""
    listings = db.query(Listing).order_by(Listing.created_at.desc()).all()
    return [serialize_listing(l, db) for l in listings]


@router.post("/listings/{listing_id}/moderate", response_model=ListingResponse)
def moderate_listing(
    listing_id: int,
    action: ModerationAction,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Suspend or restore a public marketplace listing.
    Enforces required reasoning and writes immutable records to AuditLog.
    """
    listing = db.query(Listing).filter(Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found.")

    if action.suspend:
        reason = (action.reason or "").strip()
        if not reason:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid moderation reason is required to suspend a public listing."
            )
        listing.is_suspended = True
        listing.suspension_reason = reason
        action_name = "Suspended listing"
    else:
        listing.is_suspended = False
        reason = action.reason or "Restored to active status by administrator."
        listing.suspension_reason = None
        action_name = "Restored listing"

    listing.updated_at = datetime.utcnow()

    # Create immutable audit log entry
    audit_entry = AuditLog(
        admin_id=current_admin.id,
        action=action_name,
        target_type="listing",
        target_id=str(listing.id),
        reason=reason
    )
    db.add(audit_entry)
    db.commit()
    db.refresh(listing)

    logger.info(f"Admin {current_admin.username} performed '{action_name}' on listing #{listing.id}: {reason}")
    return serialize_listing(listing, db)


@router.get("/audit-logs", response_model=List[AuditLogResponse])
def get_audit_logs(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Retrieve complete moderation audit logs with timestamps and administrative rationale."""
    logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).all()

    return [
        AuditLogResponse(
            id=l.id,
            admin_id=l.admin_id,
            admin_name=l.admin.full_name or l.admin.username if l.admin else "Admin",
            action=l.action,
            target_type=l.target_type,
            target_id=l.target_id,
            reason=l.reason,
            created_at=l.created_at
        ) for l in logs
    ]


@router.get("/stats")
def get_platform_statistics(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Platform-wide operational statistics."""
    return {
        "users_count": db.query(User).count(),
        "batches_count": db.query(Batch).count(),
        "observations_count": db.query(Observation).count(),
        "listings_count": db.query(Listing).count(),
        "active_listings_count": db.query(Listing).filter(Listing.status == "published", Listing.is_suspended == False).count(),
        "suspended_listings_count": db.query(Listing).filter(Listing.is_suspended == True).count(),
        "purchase_requests_count": db.query(PurchaseRequest).count()
    }
