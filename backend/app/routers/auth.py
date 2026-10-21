from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import UserRegister, UserLogin, UserResponse, Token
from backend.app.security import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class DemoLoginRequest(BaseModel):
    role: str  # 'farmer', 'buyer', 'admin'


DEMO_USERS = {
    "farmer": {
        "email": "farmer@harvestlens.ai",
        "username": "farmer",
        "password": "farmer123password",
        "full_name": "Ramesh Kumar (Farmer)",
        "farm_name": "Green Valley Orchards",
        "phone": "+91 98765 43210"
    },
    "buyer": {
        "email": "buyer@harvestlens.ai",
        "username": "buyer",
        "password": "buyer123password",
        "full_name": "AgroFresh Wholesale Corp",
        "farm_name": None,
        "phone": "+91 91234 56789"
    },
    "admin": {
        "email": "admin@harvestlens.ai",
        "username": "admin",
        "password": "admin123password",
        "full_name": "System Administrator",
        "farm_name": None,
        "phone": None
    }
}


def get_or_create_demo_user(db: Session, role: str) -> User:
    """Ensure demo user exists in DB with hashed password."""
    data = DEMO_USERS.get(role, DEMO_USERS["farmer"])
    user = db.query(User).filter(User.username == data["username"]).first()
    if not user:
        user = User(
            email=data["email"],
            username=data["username"],
            hashed_password=hash_password(data["password"]),
            full_name=data["full_name"],
            role=role,
            farm_name=data["farm_name"],
            phone=data["phone"]
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


@router.post("/register", response_model=Token)
def register_user(req: UserRegister, db: Session = Depends(get_db)):
    """Register a new user account (farmer, buyer, admin)."""
    # Check if email or username already exists
    if db.query(User).filter(User.email == req.email.lower()).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists."
        )
    if db.query(User).filter(User.username == req.username.lower()).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This username is already taken. Please choose another."
        )

    user = User(
        email=req.email.lower(),
        username=req.username.lower(),
        hashed_password=hash_password(req.password),
        full_name=req.full_name,
        role=req.role,
        farm_name=req.farm_name,
        phone=req.phone
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(data={
        "user_id": user.id,
        "username": user.username,
        "role": user.role
    })

    return Token(
        access_token=token,
        token_type="bearer",
        role=user.role,
        user_id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        farm_name=user.farm_name
    )


@router.post("/login", response_model=Token)
def login_user(req: UserLogin, db: Session = Depends(get_db)):
    """Authenticate with username or email and password."""
    identifier = req.username_or_email.lower().strip()
    user = db.query(User).filter(
        (User.username == identifier) | (User.email == identifier)
    ).first()

    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={
        "user_id": user.id,
        "username": user.username,
        "role": user.role
    })

    return Token(
        access_token=token,
        token_type="bearer",
        role=user.role,
        user_id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        farm_name=user.farm_name
    )


@router.post("/demo-login", response_model=Token)
def demo_login(req: DemoLoginRequest, db: Session = Depends(get_db)):
    """Quick demo login for seamless evaluation of farmer, buyer, and admin experiences."""
    role = req.role if req.role in ("farmer", "buyer", "admin") else "farmer"
    user = get_or_create_demo_user(db, role)

    token = create_access_token(data={
        "user_id": user.id,
        "username": user.username,
        "role": user.role
    })

    return Token(
        access_token=token,
        token_type="bearer",
        role=user.role,
        user_id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        farm_name=user.farm_name
    )


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Retrieve details for the currently authenticated user."""
    return current_user
