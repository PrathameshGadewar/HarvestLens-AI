import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

from backend.app.database import get_db
from backend.app.models import Message, User, Listing, PurchaseRequest
from backend.app.schemas import MessageCreate, MessageResponse
from backend.app.security import get_current_user

logger = logging.getLogger("harvestlens.messages")
router = APIRouter(prefix="/api/messages", tags=["Inquiries & Messaging"])


def serialize_message(msg: Message) -> MessageResponse:
    listing_title = msg.listing.title if msg.listing else None
    fruit = msg.listing.fruit if msg.listing else None
    return MessageResponse(
        id=msg.id,
        listing_id=msg.listing_id,
        listing_title=listing_title,
        fruit=fruit,
        request_id=msg.request_id,
        sender_id=msg.sender_id,
        receiver_id=msg.receiver_id,
        sender_name=msg.sender.full_name or msg.sender.username if msg.sender else "User",
        receiver_name=msg.receiver.full_name or msg.receiver.username if msg.receiver else "User",
        content=msg.content,
        is_read=msg.is_read,
        created_at=msg.created_at
    )


@router.post("", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def send_message(
    req: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send an inquiry or negotiation message to a buyer or seller."""
    receiver = db.query(User).filter(User.id == req.receiver_id).first()
    if not receiver:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipient user not found.")

    if req.receiver_id == current_user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot send message to yourself.")

    msg = Message(
        listing_id=req.listing_id,
        request_id=req.request_id,
        sender_id=current_user.id,
        receiver_id=req.receiver_id,
        content=req.content.strip(),
        is_read=False
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return serialize_message(msg)


@router.get("", response_model=List[MessageResponse])
def get_conversation_messages(
    other_user_id: Optional[int] = Query(None),
    listing_id: Optional[int] = Query(None),
    request_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve message history for a conversation."""
    query = db.query(Message).filter(
        or_(
            Message.sender_id == current_user.id,
            Message.receiver_id == current_user.id
        )
    )

    if other_user_id:
        query = query.filter(
            or_(
                and_(Message.sender_id == current_user.id, Message.receiver_id == other_user_id),
                and_(Message.sender_id == other_user_id, Message.receiver_id == current_user.id)
            )
        )

    if listing_id and not other_user_id:
        query = query.filter(Message.listing_id == listing_id)
    elif listing_id and other_user_id:
        query = query.filter(
            or_(
                Message.listing_id == listing_id,
                Message.listing_id.is_(None)
            )
        )

    if request_id:
        query = query.filter(Message.request_id == request_id)

    messages = query.order_by(Message.created_at.asc()).all()

    # Mark received messages as read
    for m in messages:
        if m.receiver_id == current_user.id and not m.is_read:
            m.is_read = True
    db.commit()

    return [serialize_message(m) for m in messages]


@router.get("/threads")
def get_user_threads(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List active messaging threads for the current user."""
    all_msgs = db.query(Message).filter(
        or_(
            Message.sender_id == current_user.id,
            Message.receiver_id == current_user.id
        )
    ).order_by(Message.created_at.desc()).all()

    threads = {}
    for m in all_msgs:
        other_id = m.receiver_id if m.sender_id == current_user.id else m.sender_id
        if other_id not in threads:
            other_user = m.receiver if m.sender_id == current_user.id else m.sender
            threads[other_id] = {
                "other_user_id": other_id,
                "other_user_name": other_user.full_name or other_user.username if other_user else "User",
                "other_user_role": other_user.role if other_user else "user",
                "last_message": m.content,
                "last_message_time": m.created_at.isoformat(),
                "unread_count": 1 if (m.receiver_id == current_user.id and not m.is_read) else 0
            }
        elif m.receiver_id == current_user.id and not m.is_read:
            threads[other_id]["unread_count"] += 1

    return list(threads.values())


@router.get("/inbox", response_model=List[MessageResponse])
def get_inbox_messages(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve all incoming messages received by current user."""
    msgs = db.query(Message).filter(
        Message.receiver_id == current_user.id
    ).order_by(Message.created_at.desc()).all()
    return [serialize_message(m) for m in msgs]


@router.get("/unread-count")
def get_unread_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Return count of unread messages and pending incoming purchase requests."""
    unread_msgs = db.query(Message).filter(
        Message.receiver_id == current_user.id,
        Message.is_read == False
    ).count()

    pending_reqs = 0
    if current_user.role in ("farmer", "admin"):
        pending_reqs = db.query(PurchaseRequest).join(Listing).filter(
            Listing.farmer_id == current_user.id,
            PurchaseRequest.status == "Pending"
        ).count()

    return {
        "unread_messages": unread_msgs,
        "pending_requests": pending_reqs,
        "total_unread": unread_msgs + pending_reqs
    }


@router.put("/{message_id}/read")
def mark_message_read(
    message_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark a specific message as read."""
    msg = db.query(Message).filter(Message.id == message_id, Message.receiver_id == current_user.id).first()
    if msg:
        msg.is_read = True
        db.commit()
    return {"status": "ok"}

