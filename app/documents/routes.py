from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc
from typing import List, Optional
from datetime import datetime
import os
import uuid
import mimetypes
from pathlib import Path

from app.database import get_db
from app.models import User, UserRole
from app.documents.models import Document, DocumentPermission, DocumentTemplate
from app.documents.schemas import (
    DocumentCreate, DocumentUpdate, DocumentResponse, DocumentListResponse,
    DocumentPermissionCreate, DocumentPermissionUpdate, DocumentPermissionResponse,
    DocumentTemplateCreate, DocumentTemplateUpdate, DocumentTemplateResponse,
    DocumentFilter, DocumentStats, DocumentUploadResponse
)
from app.auth.dependencies import get_current_user, require_lawyer_or_admin

router = APIRouter(prefix="/documents", tags=["Documents"])

# Configuration
UPLOAD_DIR = "uploads/documents"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.txt', '.rtf', 
    '.jpg', '.jpeg', '.png', '.gif',
    '.xls', '.xlsx', '.ppt', '.pptx'
}

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_file_info(file: UploadFile) -> dict:
    """Extract file information."""
    file_extension = Path(file.filename).suffix.lower()
    mime_type = file.content_type or mimetypes.guess_type(file.filename)[0] or 'application/octet-stream'
    
    return {
        'original_name': file.filename,
        'file_extension': file_extension,
        'mime_type': mime_type
    }

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

async def save_uploaded_file(file: UploadFile) -> tuple[str, int]:
    """Save uploaded file and return path and size."""
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix.lower()
    filename = f"{file_id}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save file
    content = await file.read()
    file_size = len(content)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    return file_path, file_size

def check_document_permission(document: Document, user: User, required_permission: str = "read") -> bool:
    """Check if user has permission to access document."""
    # Owner always has full access
    if document.uploaded_by == user.id:
        return True
    
    # Admin has full access
    if user.role == UserRole.ADMIN:
        return True
    
    # Check if user is in shared_with list
    if document.shared_with and user.id in document.shared_with:
        return True
    
    # Check explicit permissions
    permission = document.permissions.filter(
        DocumentPermission.user_id == user.id,
        DocumentPermission.expires_at.is_(None) or DocumentPermission.expires_at > datetime.utcnow()
    ).first()
    
    if permission:
        if required_permission == "read":
            return True
        elif required_permission == "write" and permission.permission in ["write", "admin"]:
            return True
        elif required_permission == "admin" and permission.permission == "admin":
            return True
    
    return False

# =====================================================
# DOCUMENT CRUD OPERATIONS
# =====================================================

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    category: str = Form(...),
    subcategory: Optional[str] = Form(None),
    case_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # JSON string
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a new document."""
    # Validate file
    validate_file(file)
    
    # Save file
    file_path, file_size = await save_uploaded_file(file)
    file_info = get_file_info(file)
    
    # Parse tags if provided
    parsed_tags = []
    if tags:
        try:
            import json
            parsed_tags = json.loads(tags)
        except:
            pass
    
    # Create document record
    document = Document(
        name=name,
        original_name=file_info['original_name'],
        file_path=file_path,
        file_size=file_size,
        mime_type=file_info['mime_type'],
        file_extension=file_info['file_extension'],
        category=category,
        subcategory=subcategory,
        case_id=case_id,
        uploaded_by=current_user.id,
        description=description,
        tags=parsed_tags
    )
    
    db.add(document)
    db.commit()
    db.refresh(document)
    
    # Load relationships for response
    document = db.query(Document).options(
        joinedload(Document.uploader)
    ).filter(Document.id == document.id).first()
    
    return DocumentUploadResponse(
        document=document,
        message="Document uploaded successfully"
    )

@router.get("/", response_model=List[DocumentListResponse])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    category: Optional[str] = None,
    status: Optional[str] = None,
    case_id: Optional[str] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List documents with filtering and pagination."""
    query = db.query(Document)
    
    # Apply role-based filtering
    if current_user.role != UserRole.ADMIN:
        # Users can only see their own documents and documents shared with them
        query = query.filter(
            or_(
                Document.uploaded_by == current_user.id,
                Document.shared_with.contains([current_user.id]),
                Document.id.in_(
                    db.query(DocumentPermission.document_id).filter(
                        DocumentPermission.user_id == current_user.id,
                        or_(
                            DocumentPermission.expires_at.is_(None),
                            DocumentPermission.expires_at > datetime.utcnow()
                        )
                    )
                )
            )
        )
    
    # Apply filters
    if category:
        query = query.filter(Document.category == category)
    if status:
        query = query.filter(Document.status == status)
    if case_id:
        query = query.filter(Document.case_id == case_id)
    if search:
        search_filter = or_(
            Document.name.ilike(f"%{search}%"),
            Document.description.ilike(f"%{search}%"),
            Document.original_name.ilike(f"%{search}%")
        )
        query = query.filter(search_filter)
    
    # Order by creation date (newest first)
    documents = query.order_by(desc(Document.created_at)).offset(skip).limit(limit).all()
    
    return documents

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific document by ID."""
    document = db.query(Document).options(
        joinedload(Document.uploader)
    ).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check permissions
    if not check_document_permission(document, current_user, "read"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return document

@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    document_update: DocumentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check permissions
    if not check_document_permission(document, current_user, "write"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Update fields
    update_data = document_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(document, field, value)
    
    document.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(document)
    
    # Load relationships for response
    document = db.query(Document).options(
        joinedload(Document.uploader)
    ).filter(Document.id == document_id).first()
    
    return document

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check permissions (only owner or admin can delete)
    if document.uploaded_by != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Delete physical file
    try:
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
    except Exception as e:
        # Log error but don't fail the request
        pass
    
    # Delete from database
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}

# =====================================================
# DOCUMENT PERMISSIONS
# =====================================================

@router.post("/{document_id}/permissions", response_model=DocumentPermissionResponse)
async def grant_document_permission(
    document_id: str,
    permission_data: DocumentPermissionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Grant permission to access a document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if user can grant permissions (owner or admin)
    if not check_document_permission(document, current_user, "admin"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if user exists
    user = db.query(User).filter(User.id == permission_data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if permission already exists
    existing = db.query(DocumentPermission).filter(
        DocumentPermission.document_id == document_id,
        DocumentPermission.user_id == permission_data.user_id
    ).first()
    
    if existing:
        # Update existing permission
        existing.permission = permission_data.permission
        existing.expires_at = permission_data.expires_at
        existing.granted_by = current_user.id
        existing.granted_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        permission = existing
    else:
        # Create new permission
        permission = DocumentPermission(
            document_id=document_id,
            user_id=permission_data.user_id,
            permission=permission_data.permission,
            expires_at=permission_data.expires_at,
            granted_by=current_user.id
        )
        db.add(permission)
        db.commit()
        db.refresh(permission)
    
    # Load relationships for response
    permission = db.query(DocumentPermission).options(
        joinedload(DocumentPermission.user),
        joinedload(DocumentPermission.granter)
    ).filter(DocumentPermission.id == permission.id).first()
    
    return permission

@router.get("/{document_id}/permissions", response_model=List[DocumentPermissionResponse])
async def get_document_permissions(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all permissions for a document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if user can view permissions
    if not check_document_permission(document, current_user, "admin"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    permissions = db.query(DocumentPermission).options(
        joinedload(DocumentPermission.user),
        joinedload(DocumentPermission.granter)
    ).filter(DocumentPermission.document_id == document_id).all()
    
    return permissions

@router.delete("/permissions/{permission_id}")
async def revoke_document_permission(
    permission_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Revoke a document permission."""
    permission = db.query(DocumentPermission).filter(DocumentPermission.id == permission_id).first()
    
    if not permission:
        raise HTTPException(status_code=404, detail="Permission not found")
    
    # Check if user can revoke permissions
    document = db.query(Document).filter(Document.id == permission.document_id).first()
    if not check_document_permission(document, current_user, "admin"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    db.delete(permission)
    db.commit()
    
    return {"message": "Permission revoked successfully"}

# =====================================================
# DOCUMENT TEMPLATES
# =====================================================

@router.post("/templates", response_model=DocumentTemplateResponse)
async def create_document_template(
    template_data: DocumentTemplateCreate,
    current_user: User = Depends(require_lawyer_or_admin()),
    db: Session = Depends(get_db)
):
    """Create a new document template."""
    template = DocumentTemplate(
        **template_data.dict(),
        created_by=current_user.id
    )
    
    db.add(template)
    db.commit()
    db.refresh(template)
    
    # Load relationships for response
    template = db.query(DocumentTemplate).options(
        joinedload(DocumentTemplate.creator)
    ).filter(DocumentTemplate.id == template.id).first()
    
    return template

@router.get("/templates", response_model=List[DocumentTemplateResponse])
async def list_document_templates(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    category: Optional[str] = None,
    is_active: bool = True,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List document templates."""
    query = db.query(DocumentTemplate).options(
        joinedload(DocumentTemplate.creator)
    )
    
    # Apply filters
    if category:
        query = query.filter(DocumentTemplate.category == category)
    if is_active is not None:
        query = query.filter(DocumentTemplate.is_active == is_active)
    
    templates = query.order_by(DocumentTemplate.name).offset(skip).limit(limit).all()
    
    return templates

@router.get("/templates/{template_id}", response_model=DocumentTemplateResponse)
async def get_document_template(
    template_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific document template."""
    template = db.query(DocumentTemplate).options(
        joinedload(DocumentTemplate.creator)
    ).filter(DocumentTemplate.id == template_id).first()
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return template

@router.put("/templates/{template_id}", response_model=DocumentTemplateResponse)
async def update_document_template(
    template_id: str,
    template_update: DocumentTemplateUpdate,
    current_user: User = Depends(require_lawyer_or_admin()),
    db: Session = Depends(get_db)
):
    """Update a document template."""
    template = db.query(DocumentTemplate).filter(DocumentTemplate.id == template_id).first()
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Update fields
    update_data = template_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(template, field, value)
    
    template.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(template)
    
    # Load relationships for response
    template = db.query(DocumentTemplate).options(
        joinedload(DocumentTemplate.creator)
    ).filter(DocumentTemplate.id == template_id).first()
    
    return template

@router.delete("/templates/{template_id}")
async def delete_document_template(
    template_id: str,
    current_user: User = Depends(require_lawyer_or_admin()),
    db: Session = Depends(get_db)
):
    """Delete a document template."""
    template = db.query(DocumentTemplate).filter(DocumentTemplate.id == template_id).first()
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    db.delete(template)
    db.commit()
    
    return {"message": "Template deleted successfully"}

# =====================================================
# DOCUMENT STATISTICS
# =====================================================

@router.get("/stats/overview", response_model=DocumentStats)
async def get_document_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get document statistics overview."""
    base_query = db.query(Document)
    
    # Apply role-based filtering
    if current_user.role != UserRole.ADMIN:
        base_query = base_query.filter(
            or_(
                Document.uploaded_by == current_user.id,
                Document.shared_with.contains([current_user.id])
            )
        )
    
    # Basic counts
    total_documents = base_query.count()
    
    # Group by category
    category_stats = base_query.with_entities(
        Document.category, func.count(Document.id)
    ).group_by(Document.category).all()
    
    by_category = {str(cat): count for cat, count in category_stats}
    
    # Group by status
    status_stats = base_query.with_entities(
        Document.status, func.count(Document.id)
    ).group_by(Document.status).all()
    
    by_status = {str(status): count for status, count in status_stats}
    
    # Calculate total size
    total_size_bytes = base_query.with_entities(
        func.sum(Document.file_size)
    ).scalar() or 0
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    # Recent uploads (last 7 days)
    from datetime import timedelta
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_uploads = base_query.filter(Document.created_at >= week_ago).count()
    
    return DocumentStats(
        total_documents=total_documents,
        by_category=by_category,
        by_status=by_status,
        total_size_mb=round(total_size_mb, 2),
        recent_uploads=recent_uploads
    )


