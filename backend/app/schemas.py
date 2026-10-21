from typing import List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


# ==========================================
# Auth & User Schemas
# ==========================================

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    user_id: int
    username: str
    email: str
    full_name: Optional[str] = None
    farm_name: Optional[str] = None


class TokenData(BaseModel):
    user_id: Optional[int] = None
    role: Optional[str] = None
    username: Optional[str] = None


class UserRegister(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    role: str = Field("farmer", pattern="^(farmer|buyer|admin)$")
    farm_name: Optional[str] = None
    phone: Optional[str] = None


class UserLogin(BaseModel):
    username_or_email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str] = None
    role: str
    farm_name: Optional[str] = None
    phone: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ==========================================
# Detection / Inference Schemas
# ==========================================

class DetectionItem(BaseModel):
    label: str
    fruit: str
    maturity: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]
    class_id: Optional[int] = None


class PredictResponse(BaseModel):
    detections: List[DetectionItem]
    image_width: int
    image_height: int
    inference_time_ms: float
    model_version: str = "YOLOv8-HarvestLens-32Class"
    annotated_image: Optional[str] = None


# ==========================================
# Observation Schemas
# ==========================================

class ObservationCreate(BaseModel):
    detections: Optional[List[DetectionItem]] = []
    notes: Optional[str] = None
    analysed: bool = True
    image_base64: Optional[str] = None  # Optional embedded data URL if uploaded directly


class ObservationResponse(BaseModel):
    id: int
    batch_id: str
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    detections: List[DetectionItem] = []
    dominant_maturity: Optional[str] = None
    notes: Optional[str] = None
    analysed: bool = True
    created_at: datetime

    class Config:
        from_attributes = True


# ==========================================
# Batch Schemas
# ==========================================

class BatchCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    fruit: str
    variety: Optional[str] = None
    plot: Optional[str] = None
    notes: Optional[str] = None
    planted_date: Optional[str] = None
    fruit_set_date: Optional[str] = None
    env_notes: Optional[str] = None
    interval_days: int = Field(7, ge=1)
    qty: float = Field(0.0, ge=0.0)
    unit: str = Field("kg")
    confirmed_category: Optional[str] = None
    initial_observation: Optional[ObservationCreate] = None


class BatchUpdate(BaseModel):
    name: Optional[str] = None
    fruit: Optional[str] = None
    variety: Optional[str] = None
    plot: Optional[str] = None
    notes: Optional[str] = None
    planted_date: Optional[str] = None
    fruit_set_date: Optional[str] = None
    env_notes: Optional[str] = None
    interval_days: Optional[int] = Field(None, ge=1)
    qty: Optional[float] = Field(None, ge=0.0)
    unit: Optional[str] = None
    confirmed_category: Optional[str] = None
    is_completed: Optional[bool] = None
    is_archived: Optional[bool] = None


class BatchResponse(BaseModel):
    id: str
    user_id: int
    name: str
    fruit: str
    variety: Optional[str] = None
    plot: Optional[str] = None
    notes: Optional[str] = None
    planted_date: Optional[str] = None
    fruit_set_date: Optional[str] = None
    env_notes: Optional[str] = None
    interval_days: int
    qty: float
    unit: str
    confirmed_category: Optional[str] = None
    is_archived: bool
    is_completed: bool
    created_at: datetime
    updated_at: datetime
    observations: List[ObservationResponse] = []
    listing: Optional[Any] = None

    class Config:
        from_attributes = True


# ==========================================
# Marketplace & Listing Schemas
# ==========================================

class ListingCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=255)
    region: str = Field(..., min_length=2, max_length=255)
    offered_qty: float = Field(..., gt=0)
    price_per_unit: float = Field(..., gt=0)
    currency: str = Field("INR")
    available_from: Optional[str] = None
    farm_name: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    description: Optional[str] = None
    publish_now: bool = True  # If true, status is 'published', else 'draft'


class ListingUpdate(BaseModel):
    title: Optional[str] = None
    region: Optional[str] = None
    offered_qty: Optional[float] = Field(None, gt=0)
    price_per_unit: Optional[float] = Field(None, gt=0)
    currency: Optional[str] = None
    available_from: Optional[str] = None
    farm_name: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(draft|published|paused|soldout)$")


class ListingResponse(BaseModel):
    id: int
    batch_id: str
    farmer_id: int
    title: str
    fruit: str
    confirmed_category: str
    variety: Optional[str] = None
    region: str
    offered_qty: float
    reserved_qty: float = 0.0
    available_qty: float = 0.0
    unit: str = "kg"
    price_per_unit: float
    currency: str = "INR"
    available_from: Optional[str] = None
    farm_name: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    description: Optional[str] = None
    status: str
    is_suspended: bool
    suspension_reason: Optional[str] = None
    published_at: Optional[datetime] = None
    created_at: datetime
    latest_image_url: Optional[str] = None

    class Config:
        from_attributes = True


# ==========================================
# Purchase Request Schemas
# ==========================================

class PurchaseRequestCreate(BaseModel):
    listing_id: int
    quantity: float = Field(..., gt=0)
    proposed_price: float = Field(..., gt=0)
    message: Optional[str] = None


class PurchaseRequestStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(Accepted|Rejected|Cancelled)$")


class PurchaseRequestResponse(BaseModel):
    id: str
    listing_id: int
    farmer_id: Optional[int] = None
    seller_name: Optional[str] = None
    buyer_id: int
    buyer_name: Optional[str] = None
    listing_title: Optional[str] = None
    fruit: Optional[str] = None
    unit: Optional[str] = None
    quantity: float
    proposed_price: float
    message: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ==========================================
# Messaging Schemas
# ==========================================

class MessageCreate(BaseModel):
    receiver_id: int
    listing_id: Optional[int] = None
    request_id: Optional[str] = None
    content: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    id: int
    listing_id: Optional[int] = None
    listing_title: Optional[str] = None
    fruit: Optional[str] = None
    request_id: Optional[str] = None
    sender_id: int
    receiver_id: int
    sender_name: Optional[str] = None
    receiver_name: Optional[str] = None
    content: str
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ==========================================
# Admin & Moderation Schemas
# ==========================================

class ModerationAction(BaseModel):
    suspend: bool
    reason: Optional[str] = None


class AuditLogResponse(BaseModel):
    id: int
    admin_id: int
    admin_name: Optional[str] = None
    action: str
    target_type: str
    target_id: str
    reason: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
