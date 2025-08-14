from sqlalchemy import Column, String, Text, DateTime, Boolean, ForeignKey, Enum, BigInteger, JSON, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
from app.models import DocumentCategory, DocumentStatus, Visibility, Permission
import uuid

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
    category = Column(Enum(DocumentCategory), nullable=False)
    subcategory = Column(String(100))
    status = Column(Enum(DocumentStatus), default=DocumentStatus.DRAFT)
    visibility = Column(Enum(Visibility), default=Visibility.PRIVATE)
    
    # Relationships
    case_id = Column(String(36), ForeignKey("cases.id"))
    uploaded_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    shared_with = Column(JSON)  # Array of user IDs
    
    # Document details
    description = Column(Text)
    tags = Column(JSON)
    version = Column(Integer, default=1)
    parent_document_id = Column(String(36), ForeignKey("documents.id"))
    
    # Security
    is_encrypted = Column(Boolean, default=True)
    encryption_key_id = Column(String(100))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="documents")
    uploader = relationship("User", foreign_keys=[uploaded_by])
    parent_document = relationship("Document", remote_side=[id])
    permissions = relationship("DocumentPermission", back_populates="document")

class DocumentPermission(Base):
    __tablename__ = "document_permissions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    permission = Column(Enum(Permission), default=Permission.READ)
    granted_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    granted_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
    
    # Relationships
    document = relationship("Document", back_populates="permissions")
    user = relationship("User", foreign_keys=[user_id])
    granter = relationship("User", foreign_keys=[granted_by])

class DocumentTemplate(Base):
    __tablename__ = "document_templates"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100), nullable=False)
    template_content = Column(Text, nullable=False)
    variables = Column(JSON)  # Template variables
    is_active = Column(Boolean, default=True)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by])


