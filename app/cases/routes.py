from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc, asc
from typing import List, Optional
from datetime import datetime, date
import uuid

from app.database import get_db
from app.models import User, UserRole
from app.models import Case, CaseAssignment, CaseUpdate, CaseMilestone
from app.cases.schemas import (
    CaseCreate, CaseUpdate as CaseUpdateSchema, CaseResponse, CaseListResponse,
    CaseAssignmentCreate, CaseAssignmentUpdate, CaseAssignmentResponse,
    CaseUpdateCreate, CaseUpdateResponse,
    CaseMilestoneCreate, CaseMilestoneUpdate, CaseMilestoneResponse,
    CaseFilter, CaseStats, CaseVerification
)
from app.auth.dependencies import get_current_user, require_lawyer_or_admin

router = APIRouter(prefix="/cases", tags=["Cases"])

# =====================================================
# CASE CRUD OPERATIONS
# =====================================================

def generate_case_number() -> str:
    """Generate a unique case number."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d")
    unique_id = str(uuid.uuid4())[:8].upper()
    return f"CASE-{timestamp}-{unique_id}"

@router.post("/", response_model=CaseResponse)
async def create_case(
    case_data: CaseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new case."""
    # Generate unique case number
    case_number = generate_case_number()
    
    # Create case
    db_case = Case(
        case_number=case_number,
        **case_data.dict(),
        tags=case_data.tags or []
    )
    
    db.add(db_case)
    db.commit()
    db.refresh(db_case)
    
    # Load relationships for response
    db_case = db.query(Case).options(
        joinedload(Case.client),
        joinedload(Case.assignments).joinedload(CaseAssignment.lawyer)
    ).filter(Case.id == db_case.id).first()
    
    return db_case

@router.get("/", response_model=List[CaseListResponse])
async def list_cases(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = None,
    category: Optional[str] = None,
    urgency: Optional[str] = None,
    client_id: Optional[str] = None,
    lawyer_id: Optional[str] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List cases with filtering and pagination."""
    query = db.query(Case)
    
    # Apply role-based filtering
    if current_user.role == UserRole.CLIENT:
        query = query.filter(Case.client_id == current_user.id)
    elif current_user.role == UserRole.LAWYER:
        query = query.join(CaseAssignment).filter(CaseAssignment.lawyer_id == current_user.id)
    
    # Apply filters
    if status:
        query = query.filter(Case.status == status)
    if category:
        query = query.filter(Case.category == category)
    if urgency:
        query = query.filter(Case.urgency == urgency)
    if client_id:
        query = query.filter(Case.client_id == client_id)
    if lawyer_id:
        query = query.join(CaseAssignment).filter(CaseAssignment.lawyer_id == lawyer_id)
    if search:
        search_filter = or_(
            Case.title.ilike(f"%{search}%"),
            Case.description.ilike(f"%{search}%"),
            Case.case_number.ilike(f"%{search}%"),
            Case.client_name.ilike(f"%{search}%")
        )
        query = query.filter(search_filter)
    
    # Order by creation date (newest first)
    query = query.order_by(desc(Case.created_at))
    
    # Apply pagination
    cases = query.offset(skip).limit(limit).all()
    
    return cases

@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific case by ID."""
    query = db.query(Case).options(
        joinedload(Case.client),
        joinedload(Case.assignments).joinedload(CaseAssignment.lawyer)
    ).filter(Case.id == case_id)
    
    # Apply role-based filtering
    if current_user.role == UserRole.CLIENT:
        query = query.filter(Case.client_id == current_user.id)
    elif current_user.role == UserRole.LAWYER:
        query = query.join(CaseAssignment).filter(CaseAssignment.lawyer_id == current_user.id)
    
    case = query.first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    return case

@router.put("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: str,
    case_update: CaseUpdateSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a case."""
    # Get case with permission check
    query = db.query(Case).filter(Case.id == case_id)
    
    # Apply role-based filtering
    if current_user.role == UserRole.CLIENT:
        query = query.filter(Case.client_id == current_user.id)
    elif current_user.role == UserRole.LAWYER:
        query = query.join(CaseAssignment).filter(CaseAssignment.lawyer_id == current_user.id)
    
    case = query.first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Update fields
    update_data = case_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(case, field, value)
    
    case.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(case)
    
    # Load relationships for response
    case = db.query(Case).options(
        joinedload(Case.client),
        joinedload(Case.assignments).joinedload(CaseAssignment.lawyer)
    ).filter(Case.id == case_id).first()
    
    return case

@router.delete("/{case_id}")
async def delete_case(
    case_id: str,
    current_user: User = Depends(require_lawyer_or_admin()),
    db: Session = Depends(get_db)
):
    """Delete a case (admin/lawyer only)."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    db.delete(case)
    db.commit()
    
    return {"message": "Case deleted successfully"}

# =====================================================
# CASE ASSIGNMENT OPERATIONS
# =====================================================

@router.post("/{case_id}/assignments", response_model=CaseAssignmentResponse)
async def assign_lawyer_to_case(
    case_id: str,
    assignment_data: CaseAssignmentCreate,
    current_user: User = Depends(require_lawyer_or_admin()),
    db: Session = Depends(get_db)
):
    """Assign a lawyer to a case."""
    # Verify case exists
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Verify lawyer exists
    lawyer = db.query(User).filter(
        User.id == assignment_data.lawyer_id,
        User.role == UserRole.LAWYER
    ).first()
    if not lawyer:
        raise HTTPException(status_code=404, detail="Lawyer not found")
    
    # Check if assignment already exists
    existing_assignment = db.query(CaseAssignment).filter(
        CaseAssignment.case_id == case_id,
        CaseAssignment.lawyer_id == assignment_data.lawyer_id,
        CaseAssignment.removed_date.is_(None)
    ).first()
    
    if existing_assignment:
        raise HTTPException(status_code=400, detail="Lawyer already assigned to this case")
    
    # Create assignment
    assignment = CaseAssignment(
        case_id=case_id,
        **assignment_data.dict(exclude={'case_id'})
    )
    
    db.add(assignment)
    
    # Update case status if first assignment
    if case.status == "submitted":
        case.status = "assigned"
        case.assigned_date = datetime.utcnow()
    
    db.commit()
    db.refresh(assignment)
    
    # Load lawyer relationship
    assignment = db.query(CaseAssignment).options(
        joinedload(CaseAssignment.lawyer)
    ).filter(CaseAssignment.id == assignment.id).first()
    
    return assignment

@router.get("/{case_id}/assignments", response_model=List[CaseAssignmentResponse])
async def get_case_assignments(
    case_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all assignments for a case."""
    # Verify access to case
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Role-based access check
    if current_user.role == UserRole.CLIENT and case.client_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == UserRole.LAWYER:
        # Check if lawyer is assigned to this case
        lawyer_assignment = db.query(CaseAssignment).filter(
            CaseAssignment.case_id == case_id,
            CaseAssignment.lawyer_id == current_user.id,
            CaseAssignment.removed_date.is_(None)
        ).first()
        if not lawyer_assignment:
            raise HTTPException(status_code=403, detail="Access denied")
    
    assignments = db.query(CaseAssignment).options(
        joinedload(CaseAssignment.lawyer)
    ).filter(
        CaseAssignment.case_id == case_id,
        CaseAssignment.removed_date.is_(None)
    ).all()
    
    return assignments

@router.put("/assignments/{assignment_id}", response_model=CaseAssignmentResponse)
async def update_case_assignment(
    assignment_id: str,
    assignment_update: CaseAssignmentUpdate,
    current_user: User = Depends(require_lawyer_or_admin()),
    db: Session = Depends(get_db)
):
    """Update a case assignment."""
    assignment = db.query(CaseAssignment).filter(CaseAssignment.id == assignment_id).first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    # Update fields
    update_data = assignment_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(assignment, field, value)
    
    db.commit()
    db.refresh(assignment)
    
    # Load lawyer relationship
    assignment = db.query(CaseAssignment).options(
        joinedload(CaseAssignment.lawyer)
    ).filter(CaseAssignment.id == assignment_id).first()
    
    return assignment

@router.delete("/assignments/{assignment_id}")
async def remove_case_assignment(
    assignment_id: str,
    current_user: User = Depends(require_lawyer_or_admin()),
    db: Session = Depends(get_db)
):
    """Remove a lawyer assignment from a case."""
    assignment = db.query(CaseAssignment).filter(CaseAssignment.id == assignment_id).first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    assignment.removed_date = datetime.utcnow()
    db.commit()
    
    return {"message": "Assignment removed successfully"}

# =====================================================
# CASE UPDATES/NOTES OPERATIONS
# =====================================================

@router.post("/{case_id}/updates", response_model=CaseUpdateResponse)
async def create_case_update(
    case_id: str,
    update_data: CaseUpdateCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add an update/note to a case."""
    # Verify access to case
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Role-based access check
    if current_user.role == UserRole.CLIENT and case.client_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == UserRole.LAWYER:
        # Check if lawyer is assigned to this case
        lawyer_assignment = db.query(CaseAssignment).filter(
            CaseAssignment.case_id == case_id,
            CaseAssignment.lawyer_id == current_user.id,
            CaseAssignment.removed_date.is_(None)
        ).first()
        if not lawyer_assignment:
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Create update
    case_update = CaseUpdate(
        case_id=case_id,
        user_id=current_user.id,
        **update_data.dict(exclude={'case_id'})
    )
    
    db.add(case_update)
    db.commit()
    db.refresh(case_update)
    
    # Load user relationship
    case_update = db.query(CaseUpdate).options(
        joinedload(CaseUpdate.user)
    ).filter(CaseUpdate.id == case_update.id).first()
    
    return case_update

@router.get("/{case_id}/updates", response_model=List[CaseUpdateResponse])
async def get_case_updates(
    case_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all updates for a case."""
    # Verify access to case (same logic as create_case_update)
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Role-based access check
    if current_user.role == UserRole.CLIENT and case.client_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == UserRole.LAWYER:
        lawyer_assignment = db.query(CaseAssignment).filter(
            CaseAssignment.case_id == case_id,
            CaseAssignment.lawyer_id == current_user.id,
            CaseAssignment.removed_date.is_(None)
        ).first()
        if not lawyer_assignment:
            raise HTTPException(status_code=403, detail="Access denied")
    
    query = db.query(CaseUpdate).options(
        joinedload(CaseUpdate.user)
    ).filter(CaseUpdate.case_id == case_id)
    
    # Filter private updates for non-lawyers
    if current_user.role == UserRole.CLIENT:
        query = query.filter(CaseUpdate.is_private == False)
    
    updates = query.order_by(desc(CaseUpdate.created_at)).offset(skip).limit(limit).all()
    
    return updates

# =====================================================
# CASE MILESTONES OPERATIONS
# =====================================================

@router.post("/{case_id}/milestones", response_model=CaseMilestoneResponse)
async def create_case_milestone(
    case_id: str,
    milestone_data: CaseMilestoneCreate,
    current_user: User = Depends(require_lawyer_or_admin()),
    db: Session = Depends(get_db)
):
    """Create a milestone for a case."""
    # Verify case exists
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    milestone = CaseMilestone(
        case_id=case_id,
        **milestone_data.dict(exclude={'case_id'})
    )
    
    db.add(milestone)
    db.commit()
    db.refresh(milestone)
    
    # Load assignee relationship
    milestone = db.query(CaseMilestone).options(
        joinedload(CaseMilestone.assignee)
    ).filter(CaseMilestone.id == milestone.id).first()
    
    return milestone

@router.get("/{case_id}/milestones", response_model=List[CaseMilestoneResponse])
async def get_case_milestones(
    case_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all milestones for a case."""
    # Verify access to case
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    milestones = db.query(CaseMilestone).options(
        joinedload(CaseMilestone.assignee)
    ).filter(CaseMilestone.case_id == case_id).order_by(CaseMilestone.due_date).all()
    
    return milestones

@router.put("/milestones/{milestone_id}", response_model=CaseMilestoneResponse)
async def update_case_milestone(
    milestone_id: str,
    milestone_update: CaseMilestoneUpdate,
    current_user: User = Depends(require_lawyer_or_admin()),
    db: Session = Depends(get_db)
):
    """Update a case milestone."""
    milestone = db.query(CaseMilestone).filter(CaseMilestone.id == milestone_id).first()
    if not milestone:
        raise HTTPException(status_code=404, detail="Milestone not found")
    
    # Update fields
    update_data = milestone_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(milestone, field, value)
    
    milestone.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(milestone)
    
    # Load assignee relationship
    milestone = db.query(CaseMilestone).options(
        joinedload(CaseMilestone.assignee)
    ).filter(CaseMilestone.id == milestone_id).first()
    
    return milestone

@router.delete("/milestones/{milestone_id}")
async def delete_case_milestone(
    milestone_id: str,
    current_user: User = Depends(require_lawyer_or_admin()),
    db: Session = Depends(get_db)
):
    """Delete a case milestone."""
    milestone = db.query(CaseMilestone).filter(CaseMilestone.id == milestone_id).first()
    if not milestone:
        raise HTTPException(status_code=404, detail="Milestone not found")
    
    db.delete(milestone)
    db.commit()
    
    return {"message": "Milestone deleted successfully"}

# =====================================================
# CASE STATISTICS AND ANALYTICS
# =====================================================

@router.get("/stats/overview", response_model=CaseStats)
async def get_case_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get case statistics overview."""
    base_query = db.query(Case)
    
    # Apply role-based filtering
    if current_user.role == UserRole.CLIENT:
        base_query = base_query.filter(Case.client_id == current_user.id)
    elif current_user.role == UserRole.LAWYER:
        base_query = base_query.join(CaseAssignment).filter(CaseAssignment.lawyer_id == current_user.id)
    
    # Basic counts
    total_cases = base_query.count()
    active_cases = base_query.filter(Case.status.in_(['submitted', 'reviewing', 'assigned', 'in-progress'])).count()
    completed_cases = base_query.filter(Case.status == 'completed').count()
    overdue_cases = base_query.filter(
        Case.deadline_date < date.today(),
        Case.status != 'completed'
    ).count()
    
    # Group by category
    category_stats = db.query(Case.category, func.count(Case.id)).group_by(Case.category)
    if current_user.role == UserRole.CLIENT:
        category_stats = category_stats.filter(Case.client_id == current_user.id)
    elif current_user.role == UserRole.LAWYER:
        category_stats = category_stats.join(CaseAssignment).filter(CaseAssignment.lawyer_id == current_user.id)
    
    by_category = {str(cat): count for cat, count in category_stats.all()}
    
    # Group by status
    status_stats = db.query(Case.status, func.count(Case.id)).group_by(Case.status)
    if current_user.role == UserRole.CLIENT:
        status_stats = status_stats.filter(Case.client_id == current_user.id)
    elif current_user.role == UserRole.LAWYER:
        status_stats = status_stats.join(CaseAssignment).filter(CaseAssignment.lawyer_id == current_user.id)
    
    by_status = {str(status): count for status, count in status_stats.all()}
    
    # Group by urgency
    urgency_stats = db.query(Case.urgency, func.count(Case.id)).group_by(Case.urgency)
    if current_user.role == UserRole.CLIENT:
        urgency_stats = urgency_stats.filter(Case.client_id == current_user.id)
    elif current_user.role == UserRole.LAWYER:
        urgency_stats = urgency_stats.join(CaseAssignment).filter(CaseAssignment.lawyer_id == current_user.id)
    
    by_urgency = {str(urgency): count for urgency, count in urgency_stats.all()}
    
    return CaseStats(
        total_cases=total_cases,
        active_cases=active_cases,
        completed_cases=completed_cases,
        overdue_cases=overdue_cases,
        by_category=by_category,
        by_status=by_status,
        by_urgency=by_urgency
    )

# =====================================================
# LEGACY COMPATIBILITY
# =====================================================

@router.post("/verify")
async def verify_case(
    verification: CaseVerification,
    current_user: User = Depends(require_lawyer_or_admin()),
    db: Session = Depends(get_db)
):
    """Legacy case verification endpoint."""
    case = db.query(Case).filter(Case.id == verification.case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Create an update instead of direct verification
    case_update = CaseUpdate(
        case_id=verification.case_id,
        user_id=current_user.id,
        title="Lawyer Verification",
        content=verification.lawyer_response,
        type="milestone"
    )
    
    db.add(case_update)
    
    # Update case status
    case.status = "verified" if verification.approved else "rejected"
    
    db.commit()
    
    return {"message": "Case verified successfully"}