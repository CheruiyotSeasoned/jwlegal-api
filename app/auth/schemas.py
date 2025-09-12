from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from app.models import UserRole, UserStatus

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: UserRole = UserRole.CLIENT
    phone: Optional[str] = None
    address: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    national_id: Optional[str] = None
    lsk_number: Optional[str] = None  # For lawyers
    specialization: Optional[str] = None  # For lawyers
    years_of_experience: Optional[int] = None  # For lawyers

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[UserRole] = None

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: UserRole
    status: UserStatus
    avatar_url: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    national_id: Optional[str] = None
    lsk_number: Optional[str] = None
    specialization: Optional[str] = None
    years_of_experience: Optional[int] = None
    profile_completed: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class UserProfileCreate(BaseModel):
    bio: Optional[str] = None
    education: Optional[str] = None
    certifications: Optional[dict] = None
    languages: Optional[dict] = None
    hourly_rate: Optional[float] = None
    availability: Optional[dict] = None
    office_address: Optional[str] = None
    practice_areas: Optional[dict] = None  # For lawyers
    client_types: Optional[dict] = None  # For lawyers

class UserProfileResponse(BaseModel):
    id: str
    user_id: str
    bio: Optional[str] = None
    education: Optional[str] = None
    certifications: Optional[dict] = None
    languages: Optional[dict] = None
    hourly_rate: Optional[float] = None
    availability: Optional[dict] = None
    office_address: Optional[str] = None
    practice_areas: Optional[dict] = None
    client_types: Optional[dict] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

