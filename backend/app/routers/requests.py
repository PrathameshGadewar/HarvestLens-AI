import uuid
import logging
from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import PurchaseRequest, Listing, User, Message
from backend.app.schemas import (
    PurchaseRequestCreate,
    PurchaseRequestStatusUpdate,
    PurchaseRequestResponse
)
from backend.app.security import get_current_user, require_buyer, require_farmer
from backend.app.routers.marketplace import calculate_reserved_qty

logger = logging.getLogger("harvestlens.requests")
router = APIRouter(prefix="/api/requests", tags=["Purchase Requests"])


def generate_request_id() -> str:
    """Generate a clean request ID like RQ-9B2F1."""
    return f"RQ-{uuid.uuid4().hex[:5].upper()}"


def serialize_request(req: PurchaseRequest) -> PurchaseRequestResponse:
    """Helper to convert PurchaseRequest model to response schema."""
    listing_title = req.listing.title if req.listing else None
    fruit = req.listing.fruit if req.listing else None
    unit = req.listing.batch.unit if req.listing and req.listing.batch else "kg"
    buyer_name = req.buyer.full_name or req.buyer.username if req.buyer else "Buyer"
    farmer_id = req.listing.farmer_id if req.listing else None
    seller_name = (req.listing.farm_name or (req.listing.farmer.full_name if req.listing and req.listing.farmer else None)) if req.listing else "Farmer"

    return PurchaseRequestResponse(
        id=req.id,
        listing_id=req.listing_id,
        farmer_id=farmer_id,
        seller_name=seller_name,
        buyer_id=req.buyer_id,
        buyer_name=buyer_name,
        listing_title=listing_title,
        fruit=fruit,
        unit=unit,
        quantity=req.quantity,
        proposed_price=req.proposed_price,
        message=req.message,
        status=req.status,
        created_at=req.created_at,
        updated_at=req.updated_at
    )


@router.post("", response_model=PurchaseRequestResponse, status_code=status.HTTP_201_CREATED)
def submit_purchase_request(
    req: PurchaseRequestCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit a purchase request / proposal for a published marketplace listing.
    Enforces quantity availability and prevents self-purchases.
    """
    listing = db.query(Listing).filter(Listing.id == req.listing_id).first()
    if not listing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found.")

    if listing.status != "published" or listing.is_suspended:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This listing is currently not available for purchase requests."
        )

    if listing.farmer_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Farmers cannot place purchase requests on their own listings."
        )

    reserved = calculate_reserved_qty(listing.id, db)
    available = max(0.0, listing.offered_qty - reserved)

    if req.quantity > available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requested quantity ({req.quantity}) exceeds currently available quantity ({available} {listing.batch.unit if listing.batch else 'units'})."
        )

    request_id = generate_request_id()
    while db.query(PurchaseRequest).filter(PurchaseRequest.id == request_id).first():
        request_id = generate_request_id()

    purchase_req = PurchaseRequest(
        id=request_id,
        listing_id=listing.id,
        buyer_id=current_user.id,
        quantity=req.quantity,
        proposed_price=req.proposed_price,
        message=req.message,
        status="Pending"
    )
    db.add(purchase_req)

    # Automatically register proposal message so it surfaces in Buyer Messages and starts a direct thread
    unit_str = listing.batch.unit if listing.batch else "kg"
    msg_body = f"Purchase Proposal #{request_id}: {req.quantity} {unit_str} @ ₹{req.proposed_price}/{unit_str}."
    if req.message and req.message.strip():
        msg_body += f"\nNote from buyer: {req.message.strip()}"

    proposal_msg = Message(
        listing_id=listing.id,
        request_id=request_id,
        sender_id=current_user.id,
        receiver_id=listing.farmer_id,
        content=msg_body,
        is_read=False
    )
    db.add(proposal_msg)
    db.commit()
    db.refresh(purchase_req)

    return serialize_request(purchase_req)


@router.get("/my", response_model=List[PurchaseRequestResponse])
def get_my_purchase_requests(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve all purchase requests submitted by the current user."""
    requests = db.query(PurchaseRequest).filter(
        PurchaseRequest.buyer_id == current_user.id
    ).order_by(PurchaseRequest.created_at.desc()).all()

    return [serialize_request(r) for r in requests]


@router.get("/listing/{listing_id}", response_model=List[PurchaseRequestResponse])
def get_requests_for_listing(
    listing_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve all purchase requests for a specific listing (Farmer only)."""
    listing = db.query(Listing).filter(Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found.")

    if current_user.role != "admin" and listing.farmer_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    requests = db.query(PurchaseRequest).filter(
        PurchaseRequest.listing_id == listing_id
    ).order_by(PurchaseRequest.created_at.desc()).all()

    return [serialize_request(r) for r in requests]


@router.get("/incoming", response_model=List[PurchaseRequestResponse])
def get_all_incoming_requests(
    current_user: User = Depends(require_farmer),
    db: Session = Depends(get_db)
):
    """Retrieve all incoming purchase requests across all of the farmer's listings."""
    requests = db.query(PurchaseRequest).join(Listing).filter(
        Listing.farmer_id == current_user.id
    ).order_by(PurchaseRequest.created_at.desc()).all()

    return [serialize_request(r) for r in requests]


@router.put("/{request_id}/status", response_model=PurchaseRequestResponse)
def update_request_status(
    request_id: str,
    status_update: PurchaseRequestStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Accept, Reject, or Cancel a purchase request.
    Farmers can Accept or Reject; Buyers can Cancel their own Pending requests.
    """
    purchase_req = db.query(PurchaseRequest).filter(PurchaseRequest.id == request_id).first()
    if not purchase_req:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found.")

    listing = purchase_req.listing
    new_status = status_update.status

    if new_status in ("Accepted", "Rejected"):
        # Farmer action
        if current_user.role != "admin" and listing.farmer_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the seller can accept or reject this purchase request."
            )

        if new_status == "Accepted":
            # Check availability
            reserved = calculate_reserved_qty(listing.id, db)
            available = max(0.0, listing.offered_qty - reserved)

            if purchase_req.quantity > available:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot accept request: exceeds remaining available quantity ({available} {listing.batch.unit if listing.batch else 'units'})."
                )

            purchase_req.status = "Accepted"
            purchase_req.updated_at = datetime.utcnow()

            # Check if sold out after this acceptance
            if (available - purchase_req.quantity) <= 0.001:
                listing.status = "soldout"
                listing.updated_at = datetime.utcnow()

            # Send notification message to buyer
            unit_name = listing.batch.unit if listing.batch else 'kg'
            status_msg = Message(
                listing_id=listing.id,
                request_id=purchase_req.id,
                sender_id=current_user.id,
                receiver_id=purchase_req.buyer_id,
                content=f"Proposal #{purchase_req.id} ACCEPTED: Your purchase proposal for {purchase_req.quantity} {unit_name} of '{listing.title}' has been accepted by the seller.",
                is_read=False
            )
            db.add(status_msg)

        elif new_status == "Rejected":
            purchase_req.status = "Rejected"
            purchase_req.updated_at = datetime.utcnow()

            # Send notification message to buyer
            status_msg = Message(
                listing_id=listing.id,
                request_id=purchase_req.id,
                sender_id=current_user.id,
                receiver_id=purchase_req.buyer_id,
                content=f"Proposal #{purchase_req.id} UPDATE: The seller was unable to accept your purchase proposal for '{listing.title}'.",
                is_read=False
            )
            db.add(status_msg)

    elif new_status == "Cancelled":
        # Buyer action
        if current_user.role != "admin" and purchase_req.buyer_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the buyer can cancel their request."
            )
        if purchase_req.status != "Pending":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel a request with status '{purchase_req.status}'."
            )

        purchase_req.status = "Cancelled"
        purchase_req.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(purchase_req)
    return serialize_request(purchase_req)
