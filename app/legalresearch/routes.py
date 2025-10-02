from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import asyncio
from datetime import datetime
import json
import aiofiles
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pathlib import Path
import shutil
import logging
import httpx
from starlette.responses import JSONResponse
from starlette.routing import Route
from fastapi import APIRouter, HTTPException, Query, Form
from fastapi import status
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from starlette.status import HTTP_401_UNAUTHORIZED
from app.auth.routes import get_current_user, User, UserRole
from app.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import re
import tempfile
import markdown2
import base64
import mimetypes
import aiofiles.os
import time

# Import your database models
from app.models import (
    ChatMessage, 
    ResearchConversation, 
    MessageAttachment, 
    ResearchRequest as DBResearchRequest, 
    AnalysisResult
)

router = APIRouter(prefix="/legal-research", tags=["Legal Research"])
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums (keeping original ones for API compatibility)
class RequestType(str, Enum):
    legal_opinion = "legal_opinion"
    case_analysis = "case_analysis"
    compliance_review = "compliance_review"
    contract_review = "contract_review"
    legislation_analysis = "legislation_analysis"

class ResearchStatus(str, Enum):
    pending = "pending"
    analyzing = "analyzing"
    researching = "researching"
    completed = "completed"
    error = "error"

class ReferenceType(str, Enum):
    case = "case"
    statute = "statute"
    regulation = "regulation"
    act = "act"
    rule = "rule"

# Pydantic Models (Response models)
class UploadedFile(BaseModel):
    id: str
    name: str
    type: str
    size: int
    content: Optional[str] = None
    file_path: Optional[str] = None

class LegalReference(BaseModel):
    id: str
    type: ReferenceType
    title: str
    citation: str
    relevance: int = Field(..., ge=0, le=100)
    summary: str
    kenyan_law_url: Optional[str] = None
    section: Optional[str] = None
    year: Optional[str] = None

class ResearchAnalysis(BaseModel):
    id: str
    summary: str
    key_findings: List[str]
    legal_position: str
    recommendations: List[str]
    risks: List[str]
    precedents: List[LegalReference]
    statutes: List[LegalReference]
    regulations: List[LegalReference]
    related_cases: List[LegalReference]
    full_analysis: str

class ResearchRequest(BaseModel):
    id: str
    type: RequestType
    title: str
    description: str
    files: List[UploadedFile]
    status: ResearchStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    analysis: Optional[ResearchAnalysis] = None
    error: Optional[str] = None
    user_id: str

class CreateResearchRequest(BaseModel):
    type: RequestType
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10, max_length=5000)

class ResearchRequestUpdate(BaseModel):
    status: Optional[ResearchStatus] = None
    analysis: Optional[ResearchAnalysis] = None
    error: Optional[str] = None

# File storage configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Helper functions
def convert_db_analysis_to_pydantic(analysis_result: AnalysisResult) -> ResearchAnalysis:
    """Convert database AnalysisResult to Pydantic ResearchAnalysis"""
    
    def convert_references(ref_data: List[dict]) -> List[LegalReference]:
        if not ref_data:
            return []
        references = []
        for ref in ref_data:
            try:
                references.append(LegalReference(**ref))
            except Exception as e:
                logger.warning(f"Failed to convert reference: {e}")
                continue
        return references
    
    return ResearchAnalysis(
        id=analysis_result.id,
        summary=analysis_result.summary or "",
        key_findings=analysis_result.key_findings or [],
        legal_position=analysis_result.legal_position or "",
        recommendations=analysis_result.recommendations or [],
        risks=analysis_result.risks or [],
        precedents=convert_references(analysis_result.precedents or []),
        statutes=convert_references(analysis_result.statutes or []),
        regulations=convert_references(analysis_result.regulations or []),
        related_cases=convert_references(analysis_result.related_cases or []),
        full_analysis=analysis_result.full_analysis or ""
    )

def get_research_status_from_conversation(conversation: ResearchConversation) -> ResearchStatus:
    """Determine research status from conversation state"""
    if not conversation.messages:
        return ResearchStatus.pending
    
    # Check if there's an analysis result
    for message in conversation.messages:
        if message.analysis_results:
            return ResearchStatus.completed
    
    # Check last message timestamp to determine if still processing
    last_message = conversation.messages[-1]
    time_diff = datetime.utcnow() - last_message.timestamp
    
    if time_diff.total_seconds() < 300:  # 5 minutes
        return ResearchStatus.analyzing
    else:
        return ResearchStatus.pending

def convert_conversation_to_research_request(
    conversation: ResearchConversation, 
    db: Session
) -> ResearchRequest:
    """Convert database conversation to API ResearchRequest format"""
    
    # Get user message (first message should be the request)
    user_message = next((m for m in conversation.messages if m.message_type == "user"), None)
    if not user_message:
        raise ValueError("No user message found in conversation")
    
    # Get attachments
    files = []
    for attachment in user_message.attachments:
        files.append(UploadedFile(
            id=attachment.id,
            name=attachment.file_name,
            type=attachment.file_type,
            size=attachment.file_size,
            file_path=attachment.file_path
        ))
    
    # Get analysis if available
    analysis = None
    analysis_result = db.query(AnalysisResult).filter(
        AnalysisResult.message_id.in_([m.id for m in conversation.messages])
    ).first()
    
    if analysis_result:
        analysis = convert_db_analysis_to_pydantic(analysis_result)
    
    # Determine request type from content (simplified mapping)
    request_type = RequestType.legal_opinion  # Default
    content_lower = user_message.content.lower()
    if "case analysis" in content_lower or "analyze case" in content_lower:
        request_type = RequestType.case_analysis
    elif "contract" in content_lower:
        request_type = RequestType.contract_review
    elif "compliance" in content_lower:
        request_type = RequestType.compliance_review
    elif "legislation" in content_lower or "statute" in content_lower:
        request_type = RequestType.legislation_analysis
    
    return ResearchRequest(
        id=conversation.id,
        type=request_type,
        title=conversation.title,
        description=user_message.content,
        files=files,
        status=get_research_status_from_conversation(conversation),
        created_at=conversation.created_at,
        completed_at=conversation.updated_at if analysis else None,
        analysis=analysis,
        user_id=conversation.user_id
    )

# API Endpoints

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Legal Research API"}

@router.post("/research/request", response_model=ResearchRequest)
async def create_research_request(
    background_tasks: BackgroundTasks,
    type: RequestType = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    files: list[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new research request"""
    if current_user.role not in [UserRole.LAWYER, UserRole.CLIENT]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to create research requests"
        )
    
    try:
        logger.info(f"Creating research request for user {current_user.email}")
        
        # Create conversation
        conversation = ResearchConversation(
            user_id=current_user.id,
            title=title,
        )
        db.add(conversation)
        db.flush()  # Get the ID
        
        # Create user message
        user_message = ChatMessage(
            conversation_id=conversation.id,
            user_id=current_user.id,
            message_type="user",
            content=description,
            request_type="analysis"
        )
        db.add(user_message)
        db.flush()  # Get the ID
        
        # Create research request record
        research_request = DBResearchRequest(
            user_id=current_user.id,
            request_type="analysis",
            content=description
        )
        db.add(research_request)
        db.flush()
        
        # Handle file uploads
        uploaded_files = []
        if files:
            allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
            for file in files:
                ext = Path(file.filename).suffix.lower()
                if ext not in allowed_extensions:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"File type not supported: {file.filename}"
                    )

                # Save file
                file_id = str(uuid.uuid4())
                safe_filename = f"{file_id}_{file.filename}"
                file_path = UPLOAD_DIR / safe_filename

                async with aiofiles.open(file_path, 'wb') as f:
                    content = await file.read()
                    await f.write(content)

                # Create attachment record
                attachment = MessageAttachment(
                    message_id=user_message.id,
                    file_name=file.filename,
                    file_path=str(file_path),
                    file_type=file.content_type or 'application/octet-stream',
                    file_size=len(content)
                )
                db.add(attachment)
                
                # Add to response list
                uploaded_files.append(UploadedFile(
                    id=attachment.id,
                    name=file.filename,
                    type=file.content_type or 'application/octet-stream',
                    size=len(content),
                    file_path=str(file_path)
                ))
        
        db.commit()
        
        # Start background processing
        background_tasks.add_task(
            process_research_request, 
            conversation.id, 
            research_request.id
        )
        
        # Return API response format
        return ResearchRequest(
            id=conversation.id,
            type=type,
            title=title,
            description=description,
            files=uploaded_files,
            status=ResearchStatus.pending,
            created_at=conversation.created_at,
            user_id=current_user.id
        )
        
    except Exception as e:
        db.rollback()
        logger.exception("Failed to create research request")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to create research request: {str(e)}"
        )

@router.get("/research/requests", response_model=List[ResearchRequest])
async def get_research_requests(
    status: Optional[ResearchStatus] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get research requests for the current user"""
    try:
        # Query conversations for the user
        query = db.query(ResearchConversation).filter(
            ResearchConversation.user_id == current_user.id
        )
        
        # Apply pagination
        conversations = query.order_by(desc(ResearchConversation.created_at)).offset(offset).limit(limit).all()
        
        # Convert to API format
        research_requests = []
        for conversation in conversations:
            try:
                request = convert_conversation_to_research_request(conversation, db)
                
                # Apply status filter if provided
                if status is None or request.status == status:
                    research_requests.append(request)
                    
            except Exception as e:
                logger.warning(f"Failed to convert conversation {conversation.id}: {e}")
                continue
        
        return research_requests
        
    except Exception as e:
        logger.exception("Failed to fetch research requests")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch research requests: {str(e)}"
        )

@router.get("/research/requests/{request_id}", response_model=ResearchRequest)
async def get_research_request(
    request_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific research request by ID"""
    try:
        conversation = db.query(ResearchConversation).filter(
            and_(
                ResearchConversation.id == request_id,
                ResearchConversation.user_id == current_user.id
            )
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Research request not found")
        
        return convert_conversation_to_research_request(conversation, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch research request")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch research request: {str(e)}"
        )

@router.delete("/research/requests/{request_id}")
async def delete_research_request(
    request_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a research request"""
    try:
        conversation = db.query(ResearchConversation).filter(
            and_(
                ResearchConversation.id == request_id,
                ResearchConversation.user_id == current_user.id
            )
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Research request not found")
        
        # Clean up files
        for message in conversation.messages:
            for attachment in message.attachments:
                if os.path.exists(attachment.file_path):
                    try:
                        os.remove(attachment.file_path)
                    except OSError:
                        logger.warning(f"Failed to delete file: {attachment.file_path}")
        
        # Delete conversation (cascading will handle related records)
        db.delete(conversation)
        db.commit()
        
        return {"message": "Research request deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception("Failed to delete research request")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete research request: {str(e)}"
        )

@router.get("/research/requests/{request_id}/analysis/export")
async def export_research_analysis(
    request_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export research analysis as JSON"""
    try:
        conversation = db.query(ResearchConversation).filter(
            and_(
                ResearchConversation.id == request_id,
                ResearchConversation.user_id == current_user.id
            )
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Research request not found")
        
        request_data = convert_conversation_to_research_request(conversation, db)
        
        if not request_data.analysis:
            raise HTTPException(status_code=400, detail="No analysis available for export")
        
        export_data = {
            "request": request_data.dict(),
            "exported_at": datetime.utcnow().isoformat()
        }
        
        return export_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to export analysis")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to export analysis: {str(e)}"
        )

@router.get("/research/statistics")
async def get_research_statistics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get research statistics for the current user"""
    try:
        # Get all conversations for user
        conversations = db.query(ResearchConversation).filter(
            ResearchConversation.user_id == current_user.id
        ).all()
        
        total_requests = len(conversations)
        completed_requests = 0
        pending_requests = 0
        error_requests = 0
        
        # Calculate stats
        for conversation in conversations:
            status = get_research_status_from_conversation(conversation)
            if status == ResearchStatus.completed:
                completed_requests += 1
            elif status in [ResearchStatus.pending, ResearchStatus.analyzing, ResearchStatus.researching]:
                pending_requests += 1
            elif status == ResearchStatus.error:
                error_requests += 1
        
        # Simple type breakdown (this could be enhanced with better categorization)
        type_breakdown = {
            "legal_opinion": total_requests // 4,
            "case_analysis": total_requests // 4,
            "compliance_review": total_requests // 4,
            "contract_review": total_requests // 4,
            "legislation_analysis": total_requests % 4
        }
        
        return {
            "total_requests": total_requests,
            "completed_requests": completed_requests,
            "pending_requests": pending_requests,
            "error_requests": error_requests,
            "completion_rate": (completed_requests / total_requests * 100) if total_requests > 0 else 0,
            "type_breakdown": type_breakdown
        }
        
    except Exception as e:
        logger.exception("Failed to get statistics")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get statistics: {str(e)}"
        )

# Background processing function
async def process_research_request(conversation_id: str, research_request_id: str):
    """Background task to process research requests with AI analysis"""
    # Get new database session for background task
    from app.database import SessionLocal
    db = SessionLocal()
    
    try:
        conversation = db.query(ResearchConversation).filter(
            ResearchConversation.id == conversation_id
        ).first()
        
        if not conversation:
            logger.error(f"Conversation {conversation_id} not found")
            return
        
        # Get the user message
        user_message = next((m for m in conversation.messages if m.message_type == "user"), None)
        if not user_message:
            logger.error(f"No user message found in conversation {conversation_id}")
            return
        
        # Simulate processing steps
        await asyncio.sleep(2)  # Analyzing
        await asyncio.sleep(3)  # Researching
        
        # Generate AI analysis
        request_data = ResearchRequest(
            id=conversation_id,
            type=RequestType.legal_opinion,  # Default
            title=conversation.title,
            description=user_message.content,
            files=[],
            status=ResearchStatus.pending,
            created_at=conversation.created_at,
            user_id=conversation.user_id
        )
        
        analysis = await generate_ai_legal_analysis(request_data)
        
        # Save analysis to database
        analysis_result = AnalysisResult(
            message_id=user_message.id,
            research_request_id=research_request_id,
            summary=analysis.summary,
            key_findings=analysis.key_findings,
            legal_position=analysis.legal_position,
            recommendations=analysis.recommendations,
            risks=analysis.risks,
            precedents=[ref.dict() for ref in analysis.precedents],
            statutes=[ref.dict() for ref in analysis.statutes],
            regulations=[ref.dict() for ref in analysis.regulations],
            related_cases=[ref.dict() for ref in analysis.related_cases],
            full_analysis=analysis.full_analysis
        )
        
        db.add(analysis_result)
        
        # Create assistant response message
        assistant_message = ChatMessage(
            conversation_id=conversation_id,
            user_id=conversation.user_id,  # System user or keep same
            message_type="assistant",
            content=analysis.summary,
            request_type="analysis"
        )
        
        db.add(assistant_message)
        
        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()
        
        db.commit()
        logger.info(f"Successfully processed research request {conversation_id}")
        
    except Exception as e:
        db.rollback()
        logger.exception(f"Failed to process research request {conversation_id}: {e}")
        
        # Try to save error status
        try:
            conversation = db.query(ResearchConversation).filter(
                ResearchConversation.id == conversation_id
            ).first()
            
            if conversation:
                error_message = ChatMessage(
                    conversation_id=conversation_id,
                    user_id=conversation.user_id,
                    message_type="assistant",
                    content=f"Analysis failed: {str(e)}",
                    request_type="analysis"
                )
                db.add(error_message)
                db.commit()
        except:
            pass
            
    finally:
        db.close()

# Keep the existing AI analysis function
async def generate_ai_legal_analysis(request: ResearchRequest) -> ResearchAnalysis:
    """Generate real AI-powered legal analysis using GPT"""
    
    # Extract text from uploaded files (you'll need to implement file reading)
    supporting_texts = []
    for f in request.files:
        if f.content:
            supporting_texts.append(f"📎 {f.name}:\n{f.content}")

    file_excerpts = "\n".join(supporting_texts) if supporting_texts else "None"

    user_context = (
        f"Request Type: {request.type}\n"
        f"Title: {request.title}\n"
        f"Description: {request.description}\n"
        f"Jurisdiction: Kenya\n"
        f"Uploaded File Excerpts: {file_excerpts}"
    )

    system_prompt = (
        "You are a senior Kenyan legal researcher. "
        "Analyze the legal issue described by the user. "
        "Return detailed analysis with references to Kenyan cases, statutes, regulations, and acts. "
        "Be precise, objective, and structured."
    )

    user_prompt = f"""Analyze the following legal research request and return JSON with these fields:
- summary
- key_findings[] (3–5 bullet points)
- legal_position (clear statement of law)
- recommendations[] (3–5 bullet points)  
- risks[] (3–5 bullet points)
- precedents[] (each with id, type, title, citation, relevance, summary, kenyan_law_url, year)
- statutes[] (each with id, type, title, citation, relevance, summary, kenyan_law_url, section)
- regulations[] (optional, same structure as statutes)
- related_cases[] (optional, same structure as precedents)
- full_analysis (a markdown report)

Request details:
{user_context}"""

    try:
        # Call GPT
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        raw_content = response.choices[0].message.content or ""

        # Remove JSON code blocks if present
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.DOTALL)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("AI returned non-JSON response: %s", raw_content[:500])
            return create_fallback_analysis_object(request, "Invalid JSON from AI")

        # Convert data to LegalReference objects
        def create_legal_reference(item_data: dict) -> LegalReference:
            ref_type = str(item_data.get("type", "case")).lower()
            
            try:
                relevance = int(item_data.get("relevance", 50))
            except (ValueError, TypeError):
                relevance = 50

            return LegalReference(
                id=str(item_data.get("id", str(uuid.uuid4()))),
                type=ReferenceType(ref_type if ref_type in [t.value for t in ReferenceType] else "case"),
                title=item_data.get("title", ""),
                citation=item_data.get("citation", ""),
                relevance=min(100, max(0, relevance)),
                summary=item_data.get("summary", ""),
                kenyan_law_url=item_data.get("kenyan_law_url"),
                section=item_data.get("section"),
                year=str(item_data.get("year")) if item_data.get("year") is not None else None
            )

        analysis = ResearchAnalysis(
            id=str(uuid.uuid4()),
            summary=data.get("summary", f"Analysis for {request.title}"),
            key_findings=data.get("key_findings", []),
            legal_position=data.get("legal_position", "Legal position analysis pending"),
            recommendations=data.get("recommendations", []),
            risks=data.get("risks", []),
            precedents=[create_legal_reference(p) for p in data.get("precedents", [])],
            statutes=[create_legal_reference(s) for s in data.get("statutes", [])],
            regulations=[create_legal_reference(r) for r in data.get("regulations", [])],
            related_cases=[create_legal_reference(c) for c in data.get("related_cases", [])],
            full_analysis=data.get(
                "full_analysis",
                f"# Legal Analysis for {request.title}\n\nAnalysis in progress..."
            )
        )

        return analysis

    except Exception as e:
        logger.exception("AI legal analysis failed: %s", e)
        return create_fallback_analysis_object(request, str(e))

def create_fallback_analysis_object(request: ResearchRequest, error_msg: str) -> ResearchAnalysis:
    """Create fallback ResearchAnalysis object when AI fails completely"""
    return ResearchAnalysis(
        id=str(uuid.uuid4()),
        summary=f"Analysis requested for {request.title}. AI processing encountered issues: {error_msg}",
        key_findings=[
            "Manual legal research recommended",
            "AI analysis was not completed successfully", 
            "Consider traditional legal research methods"
        ],
        legal_position="Legal position requires manual analysis due to AI processing limitations.",
        recommendations=[
            "Conduct manual legal research",
            "Consult with legal professionals",
            "Review primary legal sources directly"
        ],
        risks=[
            "Incomplete analysis due to technical issues",
            "Manual verification required"
        ],
        precedents=[],
        statutes=[],
        regulations=[],
        related_cases=[],
        full_analysis=f"# Legal Analysis Status\n\nAI analysis could not be completed due to: {error_msg}\n\nPlease conduct manual legal research or retry the analysis."
    )