from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import uuid

from app.cases.models import Case, CaseAssignment, CaseUpdate, CaseMilestone
from app.cases.schemas import CaseCreate, CaseUpdate as CaseUpdateSchema, CaseFilter
from app.models import User, UserRole, CaseStatus


class CaseService:
    def __init__(self, db: Session):
        self.db = db

    def generate_case_number(self) -> str:
        """Generate a unique case number."""
        timestamp = datetime.now().strftime("%Y%m%d")
        unique_id = str(uuid.uuid4())[:8].upper()
        return f"CASE-{timestamp}-{unique_id}"

    def create_case(self, case_data: CaseCreate, current_user: User) -> Case:
        """Create a new case."""
        case_number = self.generate_case_number()
        
        db_case = Case(
            case_number=case_number,
            **case_data.dict(),
            tags=case_data.tags or []
        )
        
        self.db.add(db_case)
        self.db.commit()
        self.db.refresh(db_case)
        
        return self.get_case_with_relationships(db_case.id)

    def get_case_with_relationships(self, case_id: str) -> Optional[Case]:
        """Get case with all relationships loaded."""
        return self.db.query(Case).options(
            joinedload(Case.client),
            joinedload(Case.assignments).joinedload(CaseAssignment.lawyer)
        ).filter(Case.id == case_id).first()

    def get_cases_for_user(
        self,
        user: User,
        filters: Optional[CaseFilter] = None,
        skip: int = 0,
        limit: int = 50
    ) -> List[Case]:
        """Get cases for a user with role-based filtering."""
        query = self.db.query(Case)
        
        # Apply role-based filtering
        if user.role == UserRole.CLIENT:
            query = query.filter(Case.client_id == user.id)
        elif user.role == UserRole.LAWYER:
            query = query.join(CaseAssignment).filter(
                CaseAssignment.lawyer_id == user.id,
                CaseAssignment.removed_date.is_(None)
            )
        
        # Apply additional filters
        if filters:
            if filters.status:
                query = query.filter(Case.status.in_(filters.status))
            if filters.category:
                query = query.filter(Case.category.in_(filters.category))
            if filters.urgency:
                query = query.filter(Case.urgency.in_(filters.urgency))
            if filters.priority:
                query = query.filter(Case.priority.in_(filters.priority))
            if filters.client_id:
                query = query.filter(Case.client_id == filters.client_id)
            if filters.date_from:
                query = query.filter(Case.created_at >= filters.date_from)
            if filters.date_to:
                query = query.filter(Case.created_at <= filters.date_to)
            if filters.search:
                search_filter = or_(
                    Case.title.ilike(f"%{filters.search}%"),
                    Case.description.ilike(f"%{filters.search}%"),
                    Case.case_number.ilike(f"%{filters.search}%"),
                    Case.client_name.ilike(f"%{filters.search}%")
                )
                query = query.filter(search_filter)
        
        return query.order_by(desc(Case.created_at)).offset(skip).limit(limit).all()

    def update_case(self, case_id: str, case_update: CaseUpdateSchema, user: User) -> Optional[Case]:
        """Update a case with permission checks."""
        case = self.get_case_for_user(case_id, user)
        if not case:
            return None
        
        update_data = case_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(case, field, value)
        
        case.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(case)
        
        return self.get_case_with_relationships(case_id)

    def get_case_for_user(self, case_id: str, user: User) -> Optional[Case]:
        """Get case with user permission checks."""
        query = self.db.query(Case).filter(Case.id == case_id)
        
        if user.role == UserRole.CLIENT:
            query = query.filter(Case.client_id == user.id)
        elif user.role == UserRole.LAWYER:
            query = query.join(CaseAssignment).filter(
                CaseAssignment.lawyer_id == user.id,
                CaseAssignment.removed_date.is_(None)
            )
        
        return query.first()

    def assign_lawyer(self, case_id: str, lawyer_id: str, role: str, notes: str = None) -> Optional[CaseAssignment]:
        """Assign a lawyer to a case."""
        # Verify case exists
        case = self.db.query(Case).filter(Case.id == case_id).first()
        if not case:
            return None
        
        # Verify lawyer exists
        lawyer = self.db.query(User).filter(
            User.id == lawyer_id,
            User.role == UserRole.LAWYER
        ).first()
        if not lawyer:
            return None
        
        # Check for existing assignment
        existing = self.db.query(CaseAssignment).filter(
            CaseAssignment.case_id == case_id,
            CaseAssignment.lawyer_id == lawyer_id,
            CaseAssignment.removed_date.is_(None)
        ).first()
        
        if existing:
            return None
        
        # Create assignment
        assignment = CaseAssignment(
            case_id=case_id,
            lawyer_id=lawyer_id,
            role=role,
            notes=notes
        )
        
        self.db.add(assignment)
        
        # Update case status if first assignment
        if case.status == CaseStatus.SUBMITTED:
            case.status = CaseStatus.ASSIGNED
            case.assigned_date = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(assignment)
        
        return assignment

    def get_case_stats(self, user: User) -> Dict[str, Any]:
        """Get case statistics for a user."""
        base_query = self.db.query(Case)
        
        # Apply role-based filtering
        if user.role == UserRole.CLIENT:
            base_query = base_query.filter(Case.client_id == user.id)
        elif user.role == UserRole.LAWYER:
            base_query = base_query.join(CaseAssignment).filter(
                CaseAssignment.lawyer_id == user.id,
                CaseAssignment.removed_date.is_(None)
            )
        
        # Basic counts
        total_cases = base_query.count()
        active_cases = base_query.filter(
            Case.status.in_(['submitted', 'reviewing', 'assigned', 'in-progress'])
        ).count()
        completed_cases = base_query.filter(Case.status == 'completed').count()
        overdue_cases = base_query.filter(
            Case.deadline_date < date.today(),
            Case.status != 'completed'
        ).count()
        
        # Group by category
        category_stats = base_query.with_entities(
            Case.category, func.count(Case.id)
        ).group_by(Case.category).all()
        
        by_category = {str(cat): count for cat, count in category_stats}
        
        # Group by status
        status_stats = base_query.with_entities(
            Case.status, func.count(Case.id)
        ).group_by(Case.status).all()
        
        by_status = {str(status): count for status, count in status_stats}
        
        # Group by urgency
        urgency_stats = base_query.with_entities(
            Case.urgency, func.count(Case.id)
        ).group_by(Case.urgency).all()
        
        by_urgency = {str(urgency): count for urgency, count in urgency_stats}
        
        return {
            "total_cases": total_cases,
            "active_cases": active_cases,
            "completed_cases": completed_cases,
            "overdue_cases": overdue_cases,
            "by_category": by_category,
            "by_status": by_status,
            "by_urgency": by_urgency
        }


