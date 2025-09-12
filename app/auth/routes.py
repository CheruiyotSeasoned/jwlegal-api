from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta, datetime
from app.database import get_db
from app.models import User, UserRole, UserStatus
from app.auth.schemas import UserCreate, Token, UserResponse, UserProfileCreate, UserProfileResponse
from app.auth.utils import verify_password, get_password_hash, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, GOOGLE_CLIENT_ID
from app.auth.dependencies import get_current_user
from pydantic import BaseModel
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests


router = APIRouter(prefix="/auth", tags=["Authentication"])


# Response model
class Token(BaseModel):
    access_token: str
    token_type: str

@router.post("/register", response_model=UserResponse)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        email=user_data.email,
        password_hash=hashed_password,
        name=user_data.name,
        role=user_data.role,
        phone=user_data.phone,
        address=user_data.address,
        date_of_birth=user_data.date_of_birth,
        national_id=user_data.national_id,
        lsk_number=user_data.lsk_number,
        specialization=user_data.specialization,
        years_of_experience=user_data.years_of_experience,
        status=UserStatus.ACTIVE
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@router.post("/login", response_model=Token)
def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account is not active"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role.value},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# Add this Pydantic model to your schemas
class GoogleSignupRequest(BaseModel):
    id_token: str

@router.post("/google-signup", response_model=Token)
def google_signup(request: GoogleSignupRequest, db: Session = Depends(get_db)):
    try:
        # Verify Google ID token
        id_info = id_token.verify_oauth2_token(
            request.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        email = id_info.get("email")
        name = id_info.get("name")
        picture = id_info.get("picture")
        google_id = id_info.get("sub")
        email_verified = id_info.get("email_verified")

        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Google account does not have an email"
            )

        # Find existing user by email OR google_id
        user = (
            db.query(User)
            .filter((User.email == email) | (User.google_id == google_id))
            .first()
        )

        if not user:
            # Register new Google user
            user = User(
                email=email,
                name=name,
                avatar_url=picture,
                google_id=google_id,
                auth_provider="google",
                status=UserStatus.ACTIVE if email_verified else UserStatus.PENDING,
                role=UserRole.CLIENT,
                password_hash=None,  # Not used for Google-auth users
            )

            if email_verified:
                user.email_verified_at = datetime.utcnow()

            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            # Update existing user with Google info
            updated = False
            if not user.google_id:
                user.google_id = google_id
                updated = True
            if not user.avatar_url and picture:
                user.avatar_url = picture
                updated = True
            if not user.name and name:
                user.name = name
                updated = True
            if user.auth_provider != "google":
                user.auth_provider = "google"
                updated = True
            if email_verified and not user.email_verified_at:
                user.email_verified_at = datetime.utcnow()
                updated = True

            if updated:
                db.commit()
                db.refresh(user)

        # Update last login timestamp
        user.last_login = datetime.utcnow()
        db.commit()
        db.refresh(user)

        # Generate JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email, "role": user.role.value},
            expires_delta=access_token_expires,
        )

        return {"access_token": access_token, "token_type": "bearer"}

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Google token"
        )


@router.post("/google-login")
def google_login(payload: dict, db: Session = Depends(get_db)):
    try:
        # Verify Google token
        id_info = id_token.verify_oauth2_token(
            payload["id_token"],
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        email = id_info.get("email")
        name = id_info.get("name")
        picture = id_info.get("picture")
        google_id = id_info.get("sub")
        email_verified = id_info.get("email_verified")

        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Google account has no email"
            )

        # Find user by email or google_id
        user = (
            db.query(User)
            .filter((User.email == email) | (User.google_id == google_id))
            .first()
        )

        if not user:
            # Create new user
            user = User(
                email=email,
                name=name,
                avatar_url=picture,
                google_id=google_id,
                auth_provider="google",
                role=UserRole.CLIENT,
                status=UserStatus.ACTIVE if email_verified else UserStatus.PENDING,
                password_hash=None,
            )
            if email_verified:
                user.email_verified_at = datetime.utcnow()

            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            # Update fields if needed
            if not user.google_id:
                user.google_id = google_id
            if not user.avatar_url and picture:
                user.avatar_url = picture
            if not user.name and name:
                user.name = name
            if email_verified and not user.email_verified_at:
                user.email_verified_at = datetime.utcnow()
            if user.auth_provider != "google":
                user.auth_provider = "google"

            db.commit()
            db.refresh(user)

        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        db.refresh(user)

        # Generate JWT
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email, "role": user.role.value},
            expires_delta=access_token_expires,
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "role": user.role.value,
        }

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Google token"
        )

@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    return current_user

@router.post("/profile", response_model=UserProfileResponse)
def create_user_profile(
    profile_data: UserProfileCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from app.models import UserProfile
    
    # Check if profile already exists
    existing_profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if existing_profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Profile already exists"
        )
    
    # Create new profile
    db_profile = UserProfile(
        user_id=current_user.id,
        bio=profile_data.bio,
        education=profile_data.education,
        certifications=profile_data.certifications,
        languages=profile_data.languages,
        hourly_rate=profile_data.hourly_rate,
        availability=profile_data.availability,
        office_address=profile_data.office_address,
        practice_areas=profile_data.practice_areas,
        client_types=profile_data.client_types
    )
    
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    
    return db_profile

@router.get("/profile", response_model=UserProfileResponse)
def get_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from app.models import UserProfile
    
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    
    return profile

@router.put("/profile", response_model=UserProfileResponse)
def update_user_profile(
    profile_data: UserProfileCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from app.models import UserProfile
    
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    
    # Update profile fields
    for field, value in profile_data.dict(exclude_unset=True).items():
        setattr(profile, field, value)
    
    db.commit()
    db.refresh(profile)
    
    return profile
