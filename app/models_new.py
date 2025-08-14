from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Enum, Float, Date, DECIMAL, JSON, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import enum
import uuid

# =====================================================
# ENUMS
# =====================================================

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    LAWYER = "lawyer"
    CLIENT = "client"
    JUDICIAL = "judicial"

class UserStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"
    VERIFIED = "verified"

class DeviceType(str, enum.Enum):
    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"

class CaseCategory(str, enum.Enum):
    FAMILY = "family"
    CRIMINAL = "criminal"
    CIVIL = "civil"
    BUSINESS = "business"
    REAL_ESTATE = "real-estate"
    EMPLOYMENT = "employment"
    PERSONAL_INJURY = "personal-injury"
    IMMIGRATION = "immigration"
    INTELLECTUAL_PROPERTY = "intellectual-property"
    BANKRUPTCY = "bankruptcy"
    TRAFFIC = "traffic"
    CONSTITUTIONAL = "constitutional"
    OTHER = "other"

class CaseStatus(str, enum.Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    REVIEWING = "reviewing"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    ON_HOLD = "on-hold"

class UrgencyLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class PriorityLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplexityLevel(str, enum.Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class AssignmentRole(str, enum.Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    CONSULTANT = "consultant"
    ASSOCIATE = "associate"

class UpdateType(str, enum.Enum):
    NOTE = "note"
    UPDATE = "update"
    MILESTONE = "milestone"
    DEADLINE = "deadline"
    HEARING = "hearing"
    FILING = "filing"
    CORRESPONDENCE = "correspondence"

class MilestoneType(str, enum.Enum):
    FILING = "filing"
    HEARING = "hearing"
    SETTLEMENT = "settlement"
    JUDGMENT = "judgment"
    APPEAL = "appeal"
    OTHER = "other"

class MilestoneStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"

class DocumentCategory(str, enum.Enum):
    CONTRACT = "contract"
    EVIDENCE = "evidence"
    CORRESPONDENCE = "correspondence"
    COURT_FILING = "court-filing"
    BILLING = "billing"
    RESEARCH = "research"
    LEGAL_BRIEF = "legal-brief"
    AFFIDAVIT = "affidavit"
    PLEADING = "pleading"
    OTHER = "other"

class DocumentStatus(str, enum.Enum):
    DRAFT = "draft"
    FINAL = "final"
    ARCHIVED = "archived"
    DELETED = "deleted"

class DocumentVisibility(str, enum.Enum):
    PRIVATE = "private"
    SHARED = "shared"
    PUBLIC = "public"

class PermissionLevel(str, enum.Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

class ConversationType(str, enum.Enum):
    DIRECT = "direct"
    GROUP = "group"
    CASE = "case"

class ParticipantRole(str, enum.Enum):
    PARTICIPANT = "participant"
    ADMIN = "admin"

class MessageType(str, enum.Enum):
    TEXT = "text"
    FILE = "file"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SYSTEM = "system"

class AppointmentType(str, enum.Enum):
    CONSULTATION = "consultation"
    MEETING = "meeting"
    COURT = "court"
    FOLLOW_UP = "follow-up"
    MEDIATION = "mediation"
    ARBITRATION = "arbitration"
    OTHER = "other"

class AppointmentStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"
    NO_SHOW = "no_show"

class EventType(str, enum.Enum):
    APPOINTMENT = "appointment"
    DEADLINE = "deadline"
    HEARING = "hearing"
    PERSONAL = "personal"
    OTHER = "other"

class InvoiceStatus(str, enum.Enum):
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"
    PARTIALLY_PAID = "partially_paid"

class ActivityType(str, enum.Enum):
    RESEARCH = "research"
    CONSULTATION = "consultation"
    COURT_APPEARANCE = "court_appearance"
    DOCUMENT_PREPARATION = "document_preparation"
    CORRESPONDENCE = "correspondence"
    OTHER = "other"

class BillableStatus(str, enum.Enum):
    BILLABLE = "billable"
    NON_BILLABLE = "non_billable"
    WRITE_OFF = "write_off"

class ResearchStatus(str, enum.Enum):
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class NotificationType(str, enum.Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"

# =====================================================
# CORE USER MANAGEMENT & AUTHENTICATION
# =====================================================

class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False, index=True)
    status = Column(Enum(UserStatus), default=UserStatus.PENDING, index=True)
    avatar_url = Column(String(500))
    phone = Column(String(20))
    address = Column(Text)
    date_of_birth = Column(Date)
    national_id = Column(String(20))
    lsk_number = Column(String(50), index=True)  # For lawyers
    specialization = Column(Text)  # For lawyers
    years_of_experience = Column(Integer)  # For lawyers
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime)
    email_verified_at = Column(DateTime)
    phone_verified_at = Column(DateTime)
    profile_completed = Column(Boolean, default=False)
    
    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    sessions = relationship("UserSession", back_populates="user")
    cases_as_client = relationship("Case", foreign_keys="Case.client_id", back_populates="client")
    case_assignments = relationship("CaseAssignment", back_populates="lawyer")
    case_updates = relationship("CaseUpdate", back_populates="user")
    case_milestones = relationship("CaseMilestone", foreign_keys="CaseMilestone.assigned_to", back_populates="assigned_user")
    documents = relationship("Document", foreign_keys="Document.uploaded_by", back_populates="uploader")
    conversations_created = relationship("Conversation", foreign_keys="Conversation.created_by", back_populates="creator")
    conversation_participants = relationship("ConversationParticipant", back_populates="user")
    messages = relationship("Message", foreign_keys="Message.sender_id", back_populates="sender")
    appointments_organized = relationship("Appointment", foreign_keys="Appointment.organizer_id", back_populates="organizer")
    calendar_events = relationship("CalendarEvent", back_populates="user")
    invoices_as_lawyer = relationship("Invoice", foreign_keys="Invoice.lawyer_id", back_populates="lawyer")
    time_entries = relationship("TimeEntry", foreign_keys="TimeEntry.lawyer_id", back_populates="lawyer")
    payments = relationship("Payment", foreign_keys="Payment.payer_id", back_populates="payer")
    research_submissions = relationship("ResearchSubmission", back_populates="user")
    ai_usage_logs = relationship("AIUsageLog", back_populates="user")
    notifications = relationship("Notification", back_populates="user")
    activity_logs = relationship("ActivityLog", back_populates="user")
    security_events = relationship("SecurityEvent", back_populates="user")

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    bio = Column(Text)
    education = Column(Text)
    certifications = Column(JSON)
    languages = Column(JSON)
    hourly_rate = Column(DECIMAL(10, 2))
    availability = Column(JSON)
    office_address = Column(Text)
    practice_areas = Column(JSON)  # For lawyers
    client_types = Column(JSON)  # For lawyers
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="profile")

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    device_type = Column(Enum(DeviceType), default=DeviceType.DESKTOP)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="sessions")

# =====================================================
# CASE MANAGEMENT SYSTEM
# =====================================================

class Case(Base):
    __tablename__ = "cases"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_number = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(Enum(CaseCategory), nullable=False, index=True)
    subcategory = Column(String(100))
    status = Column(Enum(CaseStatus), default=CaseStatus.DRAFT, index=True)
    urgency = Column(Enum(UrgencyLevel), default=UrgencyLevel.MEDIUM, index=True)
    priority = Column(Enum(PriorityLevel), default=PriorityLevel.MEDIUM)
    
    # Client information
    client_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    client_name = Column(String(255), nullable=False)
    client_email = Column(String(255), nullable=False)
    client_phone = Column(String(20))
    client_address = Column(Text)
    
    # Case details
    estimated_budget = Column(DECIMAL(12, 2))
    actual_cost = Column(DECIMAL(12, 2))
    court_name = Column(String(255))
    court_case_number = Column(String(100))
    filing_date = Column(Date)
    hearing_date = Column(Date)
    deadline_date = Column(Date)
    
    # Case metadata
    tags = Column(JSON)
    keywords = Column(Text)
    complexity_level = Column(Enum(ComplexityLevel), default=ComplexityLevel.MODERATE)
    
    # Timestamps
    submitted_date = Column(DateTime, default=func.now(), index=True)
    assigned_date = Column(DateTime)
    completed_date = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    client = relationship("User", foreign_keys=[client_id], back_populates="cases_as_client")
    assignments = relationship("CaseAssignment", back_populates="case")
    updates = relationship("CaseUpdate", back_populates="case")
    milestones = relationship("CaseMilestone", back_populates="case")
    documents = relationship("Document", back_populates="case")
    conversations = relationship("Conversation", back_populates="case")
    appointments = relationship("Appointment", back_populates="case")
    invoices = relationship("Invoice", back_populates="case")
    time_entries = relationship("TimeEntry", back_populates="case")
    research_submissions = relationship("ResearchSubmission", back_populates="case")

class CaseAssignment(Base):
    __tablename__ = "case_assignments"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(36), ForeignKey("cases.id"), nullable=False, index=True)
    lawyer_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    role = Column(Enum(AssignmentRole), default=AssignmentRole.PRIMARY)
    assigned_date = Column(DateTime, default=func.now())
    removed_date = Column(DateTime)
    notes = Column(Text)
    hourly_rate = Column(DECIMAL(10, 2))
    
    # Relationships
    case = relationship("Case", back_populates="assignments")
    lawyer = relationship("User", back_populates="case_assignments")

class CaseUpdate(Base):
    __tablename__ = "case_updates"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(36), ForeignKey("cases.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    type = Column(Enum(UpdateType), default=UpdateType.NOTE)
    is_private = Column(Boolean, default=False)
    attachments = Column(JSON)
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="updates")
    user = relationship("User", back_populates="case_updates")

class CaseMilestone(Base):
    __tablename__ = "case_milestones"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(36), ForeignKey("cases.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    milestone_type = Column(Enum(MilestoneType), nullable=False)
    due_date = Column(Date, index=True)
    completed_date = Column(Date)
    status = Column(Enum(MilestoneStatus), default=MilestoneStatus.PENDING, index=True)
    assigned_to = Column(String(36), ForeignKey("users.id"), index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="milestones")
    assigned_user = relationship("User", foreign_keys=[assigned_to], back_populates="case_milestones") 

# =====================================================
# DOCUMENT MANAGEMENT SYSTEM
# =====================================================

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    original_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    mime_type = Column(String(100), nullable=False)
    file_extension = Column(String(10), nullable=False)
    
    # Document metadata
    category = Column(Enum(DocumentCategory), nullable=False, index=True)
    subcategory = Column(String(100))
    status = Column(Enum(DocumentStatus), default=DocumentStatus.DRAFT, index=True)
    visibility = Column(Enum(DocumentVisibility), default=DocumentVisibility.PRIVATE)
    
    # Relationships
    case_id = Column(String(36), ForeignKey("cases.id"), index=True)
    uploaded_by = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    shared_with = Column(JSON)  # Array of user IDs
    
    # Document details
    description = Column(Text)
    tags = Column(JSON)
    version = Column(Integer, default=1)
    parent_document_id = Column(String(36), ForeignKey("documents.id"))  # For document versions
    
    # Security
    is_encrypted = Column(Boolean, default=True)
    encryption_key_id = Column(String(100))
    
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="documents")
    uploader = relationship("User", foreign_keys=[uploaded_by], back_populates="documents")
    parent_document = relationship("Document", remote_side=[id])
    permissions = relationship("DocumentPermission", back_populates="document")

class DocumentPermission(Base):
    __tablename__ = "document_permissions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    permission = Column(Enum(PermissionLevel), default=PermissionLevel.READ)
    granted_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    granted_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)
    
    # Relationships
    document = relationship("Document", back_populates="permissions")

class DocumentTemplate(Base):
    __tablename__ = "document_templates"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100), nullable=False, index=True)
    template_content = Column(Text, nullable=False)
    variables = Column(JSON)  # Template variables
    is_active = Column(Boolean, default=True, index=True)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# =====================================================
# MESSAGING & COMMUNICATION SYSTEM
# =====================================================

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255))
    type = Column(Enum(ConversationType), default=ConversationType.DIRECT)
    case_id = Column(String(36), ForeignKey("cases.id"), index=True)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    is_archived = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="conversations")
    creator = relationship("User", foreign_keys=[created_by], back_populates="conversations_created")
    participants = relationship("ConversationParticipant", back_populates="conversation")
    messages = relationship("Message", back_populates="conversation")

class ConversationParticipant(Base):
    __tablename__ = "conversation_participants"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    role = Column(Enum(ParticipantRole), default=ParticipantRole.PARTICIPANT)
    joined_at = Column(DateTime, default=func.now())
    left_at = Column(DateTime)
    is_muted = Column(Boolean, default=False)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="participants")
    user = relationship("User", back_populates="conversation_participants")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    sender_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    message_type = Column(Enum(MessageType), default=MessageType.TEXT)
    file_url = Column(String(500))
    file_name = Column(String(255))
    file_size = Column(BigInteger)
    
    # Message status
    is_edited = Column(Boolean, default=False)
    edited_at = Column(DateTime)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime)
    deleted_by = Column(String(36), ForeignKey("users.id"))
    
    # Message metadata
    reply_to_message_id = Column(String(36), ForeignKey("messages.id"))
    forwarded_from_message_id = Column(String(36), ForeignKey("messages.id"))
    
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    sender = relationship("User", foreign_keys=[sender_id], back_populates="messages")
    read_receipts = relationship("MessageRead", back_populates="message")

class MessageRead(Base):
    __tablename__ = "message_reads"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = Column(String(36), ForeignKey("messages.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    read_at = Column(DateTime, default=func.now())
    
    # Relationships
    message = relationship("Message", back_populates="read_receipts")

# =====================================================
# APPOINTMENTS & CALENDAR SYSTEM
# =====================================================

class Appointment(Base):
    __tablename__ = "appointments"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    description = Column(Text)
    case_id = Column(String(36), ForeignKey("cases.id"), index=True)
    
    # Participants
    organizer_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    attendees = Column(JSON)  # Array of user IDs
    
    # Time and location
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=False)
    timezone = Column(String(50), default="Africa/Nairobi")
    location = Column(String(255))
    is_virtual = Column(Boolean, default=False)
    meeting_url = Column(String(500))
    meeting_id = Column(String(100))  # For virtual meetings
    
    # Appointment details
    type = Column(Enum(AppointmentType), default=AppointmentType.MEETING, index=True)
    status = Column(Enum(AppointmentStatus), default=AppointmentStatus.SCHEDULED, index=True)
    priority = Column(Enum(PriorityLevel), default=PriorityLevel.MEDIUM)
    
    # Reminders
    reminder_sent = Column(Boolean, default=False)
    reminder_sent_at = Column(DateTime)
    reminder_minutes_before = Column(Integer, default=15)
    
    # Notes
    notes = Column(Text)
    outcome = Column(Text)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="appointments")
    organizer = relationship("User", foreign_keys=[organizer_id], back_populates="appointments_organized")

class CalendarEvent(Base):
    __tablename__ = "calendar_events"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    description = Column(Text)
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=False)
    is_recurring = Column(Boolean, default=False)
    recurrence_rule = Column(String(255))  # RRULE format
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    event_type = Column(Enum(EventType), default=EventType.APPOINTMENT)
    color = Column(String(7), default="#3B82F6")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="calendar_events")

# =====================================================
# BILLING & PAYMENTS SYSTEM
# =====================================================

class Invoice(Base):
    __tablename__ = "invoices"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    invoice_number = Column(String(50), unique=True, nullable=False, index=True)
    case_id = Column(String(36), ForeignKey("cases.id"), nullable=False, index=True)
    client_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    lawyer_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    
    # Invoice details
    title = Column(String(255), nullable=False)
    description = Column(Text)
    subtotal = Column(DECIMAL(12, 2), nullable=False)
    tax_amount = Column(DECIMAL(12, 2), default=0)
    discount_amount = Column(DECIMAL(12, 2), default=0)
    total_amount = Column(DECIMAL(12, 2), nullable=False)
    
    # Status and dates
    status = Column(Enum(InvoiceStatus), default=InvoiceStatus.DRAFT, index=True)
    issue_date = Column(Date, nullable=False)
    due_date = Column(Date, nullable=False, index=True)
    paid_date = Column(Date)
    
    # Payment details
    payment_method = Column(String(50))
    payment_reference = Column(String(100))
    transaction_id = Column(String(100))
    
    # Currency
    currency = Column(String(3), default="KES")
    exchange_rate = Column(DECIMAL(10, 6), default=1.0)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="invoices")
    client = relationship("User", foreign_keys=[client_id])
    lawyer = relationship("User", foreign_keys=[lawyer_id], back_populates="invoices_as_lawyer")
    items = relationship("InvoiceItem", back_populates="invoice")

class InvoiceItem(Base):
    __tablename__ = "invoice_items"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    invoice_id = Column(String(36), ForeignKey("invoices.id"), nullable=False, index=True)
    description = Column(String(255), nullable=False)
    quantity = Column(DECIMAL(10, 2), nullable=False)
    unit_price = Column(DECIMAL(10, 2), nullable=False)
    total_price = Column(DECIMAL(12, 2), nullable=False)
    tax_rate = Column(DECIMAL(5, 2), default=0)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    invoice = relationship("Invoice", back_populates="items")

class TimeEntry(Base):
    __tablename__ = "time_entries"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(36), ForeignKey("cases.id"), nullable=False, index=True)
    lawyer_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    description = Column(String(255), nullable=False)
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    hourly_rate = Column(DECIMAL(10, 2), nullable=False)
    total_amount = Column(DECIMAL(12, 2), nullable=False)
    is_billable = Column(Boolean, default=True)
    invoice_id = Column(String(36), ForeignKey("invoices.id"))
    
    # Time entry details
    activity_type = Column(Enum(ActivityType), default=ActivityType.OTHER)
    billable_status = Column(Enum(BillableStatus), default=BillableStatus.BILLABLE)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="time_entries")
    lawyer = relationship("User", foreign_keys=[lawyer_id], back_populates="time_entries")

class Payment(Base):
    __tablename__ = "payments"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    invoice_id = Column(String(36), ForeignKey("invoices.id"), nullable=False, index=True)
    payer_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    amount = Column(DECIMAL(12, 2), nullable=False)
    payment_method = Column(String(50), nullable=False)
    payment_reference = Column(String(100))
    transaction_id = Column(String(100), unique=True)
    status = Column(String(20), default="pending")
    payment_date = Column(DateTime, default=func.now())
    notes = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    payer = relationship("User", foreign_keys=[payer_id], back_populates="payments")

# =====================================================
# AI & RESEARCH SYSTEM
# =====================================================

class ResearchSubmission(Base):
    __tablename__ = "research_submissions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    case_id = Column(String(36), ForeignKey("cases.id"), index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    research_type = Column(String(50), nullable=False)
    priority = Column(Enum(PriorityLevel), default=PriorityLevel.MEDIUM, index=True)
    status = Column(Enum(ResearchStatus), default=ResearchStatus.SUBMITTED, index=True)
    deadline = Column(DateTime)
    assigned_to = Column(String(36), ForeignKey("users.id"))
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="research_submissions")
    case = relationship("Case", back_populates="research_submissions")
    responses = relationship("ResearchResponse", back_populates="submission")

class ResearchResponse(Base):
    __tablename__ = "research_responses"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    submission_id = Column(String(36), ForeignKey("research_submissions.id"), nullable=False, index=True)
    researcher_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSON)
    attachments = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    submission = relationship("ResearchSubmission", back_populates="responses")

class AIUsageLog(Base):
    __tablename__ = "ai_usage_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    model_name = Column(String(50), nullable=False)
    tokens_used = Column(Integer, nullable=False)
    cost = Column(DECIMAL(10, 4), nullable=False)
    request_data = Column(JSON)
    response_data = Column(JSON)
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    user = relationship("User", back_populates="ai_usage_logs")

class LegalPrecedent(Base):
    __tablename__ = "legal_precedents"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    case_number = Column(String(100))
    court = Column(String(255))
    judge = Column(String(255))
    decision_date = Column(Date)
    summary = Column(Text)
    full_text = Column(Text)
    keywords = Column(JSON)
    category = Column(String(100), index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# =====================================================
# NOTIFICATION SYSTEM
# =====================================================

class Notification(Base):
    __tablename__ = "notifications"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    type = Column(Enum(NotificationType), default=NotificationType.IN_APP)
    is_read = Column(Boolean, default=False, index=True)
    read_at = Column(DateTime)
    action_url = Column(String(500))
    metadata = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="notifications")

class NotificationTemplate(Base):
    __tablename__ = "notification_templates"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    type = Column(Enum(NotificationType), nullable=False)
    subject = Column(String(255))
    body = Column(Text, nullable=False)
    variables = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# =====================================================
# ANALYTICS & REPORTING
# =====================================================

class AnalyticsData(Base):
    __tablename__ = "analytics_data"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(DECIMAL(15, 4), nullable=False)
    metric_date = Column(Date, nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), index=True)
    case_id = Column(String(36), ForeignKey("cases.id"), index=True)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=func.now())

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    report_type = Column(String(50), nullable=False)
    parameters = Column(JSON)
    generated_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    file_path = Column(String(500))
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)

# =====================================================
# SYSTEM & CONFIGURATION
# =====================================================

class SystemSetting(Base):
    __tablename__ = "system_settings"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    setting_key = Column(String(100), unique=True, nullable=False)
    setting_value = Column(Text, nullable=False)
    setting_type = Column(String(20), default="string")
    description = Column(Text)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    preference_key = Column(String(100), nullable=False)
    preference_value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# =====================================================
# AUDIT & SECURITY
# =====================================================

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(36))
    details = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    user = relationship("User", back_populates="activity_logs")

class SecurityEvent(Base):
    __tablename__ = "security_events"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), index=True)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), default="medium")
    description = Column(Text, nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="security_events")

class APIAccessLog(Base):
    __tablename__ = "api_access_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), index=True)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Integer)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    request_data = Column(JSON)
    response_data = Column(JSON)
    created_at = Column(DateTime, default=func.now(), index=True) 