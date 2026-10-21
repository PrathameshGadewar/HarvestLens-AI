import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from backend.app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(String(50), default="farmer", nullable=False)  # "farmer", "buyer", "admin"
    farm_name = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    # Relationships
    batches = relationship("Batch", back_populates="owner", cascade="all, delete-orphan")
    listings = relationship("Listing", back_populates="farmer", cascade="all, delete-orphan")
    purchase_requests = relationship("PurchaseRequest", back_populates="buyer", cascade="all, delete-orphan")
    sent_messages = relationship("Message", foreign_keys="[Message.sender_id]", back_populates="sender")
    received_messages = relationship("Message", foreign_keys="[Message.receiver_id]", back_populates="receiver")
    audit_logs = relationship("AuditLog", back_populates="admin")


class Batch(Base):
    __tablename__ = "batches"

    id = Column(String(64), primary_key=True, index=True)  # e.g. "HL-9B2F1"
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    fruit = Column(String(100), nullable=False)
    variety = Column(String(100), nullable=True)
    plot = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    planted_date = Column(String(50), nullable=True)
    fruit_set_date = Column(String(50), nullable=True)
    env_notes = Column(Text, nullable=True)
    interval_days = Column(Integer, default=7, nullable=False)
    qty = Column(Float, default=0.0, nullable=False)
    unit = Column(String(50), default="kg", nullable=False)
    confirmed_category = Column(String(50), nullable=True)  # "Unripe", "Ripe", "Overripe", "Rotten"
    is_archived = Column(Boolean, default=False, nullable=False)
    is_completed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    # Relationships
    owner = relationship("User", back_populates="batches")
    observations = relationship("Observation", back_populates="batch", cascade="all, delete-orphan", order_by="Observation.created_at")
    listing = relationship("Listing", back_populates="batch", uselist=False, cascade="all, delete-orphan")


class Observation(Base):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String(64), ForeignKey("batches.id", ondelete="CASCADE"), nullable=False, index=True)
    image_path = Column(String(500), nullable=True)
    detections = Column(Text, default="[]", nullable=False)  # JSON serialized array of detections
    dominant_maturity = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)
    analysed = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    # Relationships
    batch = relationship("Batch", back_populates="observations")


class Listing(Base):
    __tablename__ = "listings"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String(64), ForeignKey("batches.id", ondelete="CASCADE"), unique=True, nullable=False)
    farmer_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    fruit = Column(String(100), nullable=False)
    confirmed_category = Column(String(50), nullable=False)  # "Unripe", "Ripe", "Overripe", "Rotten"
    variety = Column(String(100), nullable=True)
    region = Column(String(255), nullable=False)
    offered_qty = Column(Float, nullable=False)
    price_per_unit = Column(Float, nullable=False)
    currency = Column(String(20), default="INR", nullable=False)
    available_from = Column(String(50), nullable=True)
    farm_name = Column(String(255), nullable=True)
    contact_phone = Column(String(50), nullable=True)
    contact_email = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(String(50), default="draft", nullable=False)  # "draft", "published", "paused", "soldout"
    is_suspended = Column(Boolean, default=False, nullable=False)
    suspension_reason = Column(Text, nullable=True)
    published_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    # Relationships
    batch = relationship("Batch", back_populates="listing")
    farmer = relationship("User", back_populates="listings")
    purchase_requests = relationship("PurchaseRequest", back_populates="listing", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="listing", cascade="all, delete-orphan")


class PurchaseRequest(Base):
    __tablename__ = "purchase_requests"

    id = Column(String(64), primary_key=True, index=True)  # e.g. "RQ-91XA3"
    listing_id = Column(Integer, ForeignKey("listings.id", ondelete="CASCADE"), nullable=False, index=True)
    buyer_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    quantity = Column(Float, nullable=False)
    proposed_price = Column(Float, nullable=False)
    message = Column(Text, nullable=True)
    status = Column(String(50), default="Pending", nullable=False)  # "Pending", "Accepted", "Rejected", "Cancelled"
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    # Relationships
    listing = relationship("Listing", back_populates="purchase_requests")
    buyer = relationship("User", back_populates="purchase_requests")
    messages = relationship("Message", back_populates="request", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(Integer, ForeignKey("listings.id", ondelete="CASCADE"), nullable=True, index=True)
    request_id = Column(String(64), ForeignKey("purchase_requests.id", ondelete="CASCADE"), nullable=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    receiver_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    # Relationships
    listing = relationship("Listing", back_populates="messages")
    request = relationship("PurchaseRequest", back_populates="messages")
    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_messages")
    receiver = relationship("User", foreign_keys=[receiver_id], back_populates="received_messages")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    action = Column(String(100), nullable=False)
    target_type = Column(String(50), nullable=False)
    target_id = Column(String(64), nullable=False)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    # Relationships
    admin = relationship("User", back_populates="audit_logs")
