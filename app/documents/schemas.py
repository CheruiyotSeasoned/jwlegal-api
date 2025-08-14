from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models import DocumentCategory, DocumentStatus, Visibility, Permission

class UserBasic(BaseModel):
    id: str
    name: str
    email: str

class DocumentBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: DocumentCategory
    subcategory: Optional[str] = None
    visibility: Visibility = Visibility.PRIVATE
    tags: Optional[List[str]] = None

class DocumentCreate(DocumentBase):
    case_id: Optional[str] = None
    shared_with: Optional[List[str]] = None

class DocumentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[DocumentCategory] = None
    subcategory: Optional[str] = None
    status: Optional[DocumentStatus] = None
    visibility: Optional[Visibility] = None
    tags: Optional[List[str]] = None
    shared_with: Optional[List[str]] = None

class DocumentResponse(DocumentBase):
    id: str
    original_name: str
    file_path: str
    file_size: int
    mime_type: str
    file_extension: str
    status: DocumentStatus
    case_id: Optional[str] = None
    uploaded_by: str
    shared_with: Optional[List[str]] = None
    version: int
    parent_document_id: Optional[str] = None
    is_encrypted: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Relationships
    uploader: Optional[UserBasic] = None
    
    class Config:
        from_attributes = True

class DocumentListResponse(BaseModel):
    id: str
    name: str
    original_name: str
    category: DocumentCategory
    status: DocumentStatus
    file_size: int
    mime_type: str
    uploaded_by: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# Document Permission schemas
class DocumentPermissionCreate(BaseModel):
    document_id: str
    user_id: str
    permission: Permission = Permission.READ
    expires_at: Optional[datetime] = None

class DocumentPermissionUpdate(BaseModel):
    permission: Optional[Permission] = None
    expires_at: Optional[datetime] = None

class DocumentPermissionResponse(BaseModel):
    id: str
    document_id: str
    user_id: str
    permission: Permission
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime] = None
    user: Optional[UserBasic] = None
    granter: Optional[UserBasic] = None
    
    class Config:
        from_attributes = True

# Document Template schemas
class DocumentTemplateBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: str
    template_content: str
    variables: Optional[Dict[str, Any]] = None

class DocumentTemplateCreate(DocumentTemplateBase):
    pass

class DocumentTemplateUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = None
    template_content: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class DocumentTemplateResponse(DocumentTemplateBase):
    id: str
    is_active: bool
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    creator: Optional[UserBasic] = None
    
    class Config:
        from_attributes = True

# Upload response
class DocumentUploadResponse(BaseModel):
    document: DocumentResponse
    upload_url: Optional[str] = None
    message: str

# Search and filter schemas
class DocumentFilter(BaseModel):
    category: Optional[List[DocumentCategory]] = None
    status: Optional[List[DocumentStatus]] = None
    visibility: Optional[List[Visibility]] = None
    case_id: Optional[str] = None
    uploaded_by: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    search: Optional[str] = None
    tags: Optional[List[str]] = None

class DocumentStats(BaseModel):
    total_documents: int
    by_category: Dict[str, int]
    by_status: Dict[str, int]
    total_size_mb: float
    recent_uploads: int


