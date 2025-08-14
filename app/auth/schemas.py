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

# ===============================
# app/auth/utils.py
# ===============================
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from decouple import config

SECRET_KEY = config("SECRET_KEY")
ALGORITHM = config("ALGORITHM", default="HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = config("ACCESS_TOKEN_EXPIRE_MINUTES", default=30, cast=int)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        role: str = payload.get("role")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email, role=role)
        return token_data
    except JWTError:
        raise credentials_exception