from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, UserRole, UserStatus
from app.auth.utils import verify_token

bearer_scheme = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db)
):
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = verify_token(token, credentials_exception)
    user = db.query(User).filter(User.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    return user

def check_usage_limits(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Example logic: check if user's usage exceeds plan limit
    if current_user.monthly_requests_used >= current_user.monthly_request_limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usage limit exceeded. Upgrade your plan to continue using the service."
        )
    return current_user

def require_role(allowed_roles: list[UserRole]):
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker


def require_active_user():
    def active_checker(current_user: User = Depends(get_current_user)):
        if current_user.status != UserStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is not active"
            )
        return current_user
    return active_checker


# Convenience wrappers
def require_admin():
    return require_role([UserRole.ADMIN])


def require_lawyer_or_admin():
    return require_role([UserRole.LAWYER, UserRole.ADMIN])


def require_judge_or_admin():
    return require_role([UserRole.JUDICIAL, UserRole.ADMIN])


def require_client_or_lawyer():
    return require_role([UserRole.CLIENT, UserRole.LAWYER, UserRole.ADMIN])
