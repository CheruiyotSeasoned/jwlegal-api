from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from app.models import CaseCategory, CaseStatus, UrgencyLevel, PriorityLevel, ComplexityLevel, AssignmentRole, MilestoneType, MilestoneStatus, UpdateType

# Base schemas
class CaseBase(BaseModel):
    title: str = Field(..., min_length=5, max_length=255)
    description: Optional[str] = None
    category: CaseCategory
    subcategory: Optional[str] = None
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    priority: PriorityLevel = PriorityLevel.MEDIUM
    complexity_level: ComplexityLevel = ComplexityLevel.MODERATE
    
    # Client information
    client_name: str = Field(..., max_length=255)
    client_email: str = Field(..., max_length=255)
    client_phone: Optional[str] = None
    client_address: Optional[str] = None
    
    # Case details
    estimated_budget: Optional[Decimal] = None
    court_name: Optional[str] = None
    court_case_number: Optional[str] = None
    filing_date: Optional[date] = None
    hearing_date: Optional[date] = None
    deadline_date: Optional[date] = None
    
    # Metadata
    tags: Optional[List[str]] = None
    keywords: Optional[str] = None

class CaseCreate(CaseBase):
    client_id: str

class CaseUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=5, max_length=255)
    description: Optional[str] = None
    category: Optional[CaseCategory] = None
    subcategory: Optional[str] = None
    status: Optional[CaseStatus] = None
    urgency: Optional[UrgencyLevel] = None
    priority: Optional[PriorityLevel] = None
    complexity_level: Optional[ComplexityLevel] = None
    
    # Client information
    client_name: Optional[str] = None
    client_email: Optional[str] = None
    client_phone: Optional[str] = None
    client_address: Optional[str] = None
    
    # Case details
    estimated_budget: Optional[Decimal] = None
    actual_cost: Optional[Decimal] = None
    court_name: Optional[str] = None
    court_case_number: Optional[str] = None
    filing_date: Optional[date] = None
    hearing_date: Optional[date] = None
    deadline_date: Optional[date] = None
    
    # Metadata
    tags: Optional[List[str]] = None
    keywords: Optional[str] = None

class UserBasic(BaseModel):
    id: str
    name: str
    email: str
    role: str

class CaseAssignmentResponse(BaseModel):
    id: str
    case_id: str
    lawyer_id: str
    role: AssignmentRole
    assigned_date: datetime
    removed_date: Optional[datetime] = None
    notes: Optional[str] = None
    hourly_rate: Optional[Decimal] = None
    lawyer: Optional[UserBasic] = None
    
    class Config:
        from_attributes = True

class CaseResponse(CaseBase):
    id: str
    case_number: str
    status: CaseStatus
    actual_cost: Optional[Decimal] = None
    submitted_date: datetime
    assigned_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Relationships
    client: Optional[UserBasic] = None
    assignments: Optional[List[CaseAssignmentResponse]] = None
    
    class Config:
        from_attributes = True

class CaseListResponse(BaseModel):
    id: str
    case_number: str
    title: str
    category: CaseCategory
    status: CaseStatus
    urgency: UrgencyLevel
    priority: PriorityLevel
    client_name: str
    submitted_date: datetime
    assigned_date: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Case Assignment schemas
class CaseAssignmentCreate(BaseModel):
    case_id: str
    lawyer_id: str
    role: AssignmentRole = AssignmentRole.PRIMARY
    notes: Optional[str] = None
    hourly_rate: Optional[Decimal] = None

class CaseAssignmentUpdate(BaseModel):
    role: Optional[AssignmentRole] = None
    notes: Optional[str] = None
    hourly_rate: Optional[Decimal] = None

# Case Update schemas
class CaseUpdateCreate(BaseModel):
    case_id: str
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    type: UpdateType = UpdateType.NOTE
    is_private: bool = False
    attachments: Optional[List[Dict[str, Any]]] = None

class CaseUpdateResponse(BaseModel):
    id: str
    case_id: str
    user_id: str
    title: str
    content: str
    type: UpdateType
    is_private: bool
    attachments: Optional[List[Dict[str, Any]]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    user: Optional[UserBasic] = None
    
    class Config:
        from_attributes = True

# Case Milestone schemas
class CaseMilestoneCreate(BaseModel):
    case_id: str
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    milestone_type: MilestoneType
    due_date: Optional[date] = None
    assigned_to: Optional[str] = None

class CaseMilestoneUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    milestone_type: Optional[MilestoneType] = None
    due_date: Optional[date] = None
    completed_date: Optional[date] = None
    status: Optional[MilestoneStatus] = None
    assigned_to: Optional[str] = None

class CaseMilestoneResponse(BaseModel):
    id: str
    case_id: str
    title: str
    description: Optional[str] = None
    milestone_type: MilestoneType
    due_date: Optional[date] = None
    completed_date: Optional[date] = None
    status: MilestoneStatus
    assigned_to: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    assignee: Optional[UserBasic] = None
    
    class Config:
        from_attributes = True

# Search and filter schemas
class CaseFilter(BaseModel):
    status: Optional[List[CaseStatus]] = None
    category: Optional[List[CaseCategory]] = None
    urgency: Optional[List[UrgencyLevel]] = None
    priority: Optional[List[PriorityLevel]] = None
    client_id: Optional[str] = None
    lawyer_id: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    search: Optional[str] = None

class CaseStats(BaseModel):
    total_cases: int
    active_cases: int
    completed_cases: int
    overdue_cases: int
    by_category: Dict[str, int]
    by_status: Dict[str, int]
    by_urgency: Dict[str, int]

# Legacy compatibility
class CaseVerification(BaseModel):
    case_id: str
    lawyer_response: str = Field(..., min_length=10, max_length=2000)
    approved: bool