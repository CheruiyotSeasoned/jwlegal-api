from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models import User, Case, UserRole
from app.auth.dependencies import require_admin, require_lawyer_or_admin

router = APIRouter(prefix="/admin", tags=["Admin Tools"])

@router.get("/users")
def get_all_users(
    current_user: User = Depends(require_admin()),
    db: Session = Depends(get_db)
):
    users = db.query(User).all()
    return [
        {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "is_active": user.is_active,
            "monthly_requests_used": user.monthly_requests_used,
            "lawyer_rating": user.lawyer_rating,
            "created_at": user.created_at
        }
        for user in users
    ]

@router.get("/reviews")
def get_pending_reviews(
    current_user: User = Depends(require_admin()),
    db: Session = Depends(get_db)
):
    cases = db.query(Case).filter(
        Case.status == "verified",
        Case.flagged_by_admin == False
    ).all()
    
    return [
        {
            "case_id": case.id,
            "title": case.title,
            "category": case.category,
            "user_email": case.user.email,
            "lawyer_email": case.verifying_lawyer.email if case.verifying_lawyer else None,
            "lawyer_response": case.lawyer_response,
            "verified_at": case.verified_at,
            "created_at": case.created_at
        }
        for case in cases
    ]

@router.post("/rate-lawyer")
def rate_lawyer(
    lawyer_id: int,
    rating: float,
    case_id: int,
    notes: str = None,
    current_user: User = Depends(require_admin()),
    db: Session = Depends(get_db)
):
    if not (1.0 <= rating <= 5.0):
        raise HTTPException(status_code=400, detail="Rating must be between 1.0 and 5.0")
    
    lawyer = db.query(User).filter(
        User.id == lawyer_id,
        User.role == UserRole.LAWYER
    ).first()
    
    if not lawyer:
        raise HTTPException(status_code=404, detail="Lawyer not found")
    
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Update lawyer's overall rating (simple average for now)
    total_ratings = lawyer.verified_cases_count
    current_total = lawyer.lawyer_rating * total_ratings
    new_average = (current_total + rating) / (total_ratings + 1) if total_ratings > 0 else rating
    
    lawyer.lawyer_rating = round(new_average, 2)
    
    # Add admin notes to the case
    if notes:
        case.admin_notes = notes
    
    db.commit()
    
    return {
        "message": "Lawyer rated successfully",
        "new_rating": lawyer.lawyer_rating
    }

@router.patch("/ban-user")
def ban_user(
    user_id: int,
    reason: str,
    current_user: User = Depends(require_admin()),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.role == UserRole.SUPER_ADMIN:
        raise HTTPException(status_code=400, detail="Cannot ban super admin")
    
    user.is_active = False
    
    # Log the ban reason (you might want a separate table for this)
    # For now, we'll use admin_notes in related cases
    user_cases = db.query(Case).filter(Case.user_id == user_id).all()
    for case in user_cases:
        case.admin_notes = f"User banned: {reason}"
    
    db.commit()
    
    return {"message": f"User {user.email} has been banned", "reason": reason}

@router.patch("/flag-case")
def flag_case(
    case_id: int,
    reason: str,
    current_user: User = Depends(require_admin()),
    db: Session = Depends(get_db)
):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    case.flagged_by_admin = True
    case.admin_notes = reason
    
    db.commit()
    
    return {"message": "Case flagged successfully"}
