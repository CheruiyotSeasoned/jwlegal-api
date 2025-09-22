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
import re
import tempfile
import markdown2
import base64
import mimetypes
import aiofiles.os
import time


router = APIRouter(prefix="/legal-research", tags=["Legal  Research"])
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums
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

# Pydantic Models
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

class CreateResearchRequest(BaseModel):
    type: RequestType
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10, max_length=5000)

class ResearchRequestUpdate(BaseModel):
    status: Optional[ResearchStatus] = None
    analysis: Optional[ResearchAnalysis] = None
    error: Optional[str] = None

# In-memory storage (replace with database in production)
research_requests: Dict[str, ResearchRequest] = {}
uploaded_files: Dict[str, UploadedFile] = {}

# File storage configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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
    current_user: User = Depends(get_current_user)  # ✅ correct place
):
    if current_user.role not in [UserRole.LAWYER, UserRole.CLIENT]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to create research requests"
        )
    try:
        # Create research request
        logger.info(f"Creating research request for user {current_user.email}")
        logger.info(f"Creating research request for user id {current_user.id}")
        request_id = str(uuid.uuid4())
        research_request = ResearchRequest(
            id=request_id,
            type=type,
            title=title,
            description=description,
            files=[],
            status=ResearchStatus.pending,
            created_at=datetime.now()
        )
        research_requests[request_id] = research_request

        # Handle any attached files
        if files:
            allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
            for file in files:
                ext = Path(file.filename).suffix.lower()
                if ext not in allowed_extensions:
                    raise HTTPException(status_code=400, detail=f"File type not supported: {file.filename}")

                # Save file
                file_id = str(uuid.uuid4())
                safe_filename = f"{file_id}_{file.filename}"
                file_path = UPLOAD_DIR / safe_filename

                async with aiofiles.open(file_path, 'wb') as f:
                    content = await file.read()
                    await f.write(content)

                # Read text content if .txt
                file_content = None
                if ext == '.txt':
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        file_content = await f.read()

                uploaded_file = UploadedFile(
                    id=file_id,
                    name=file.filename,
                    type=file.content_type or 'application/octet-stream',
                    size=len(content),
                    content=file_content,
                    file_path=str(file_path)
                )

                uploaded_files[file_id] = uploaded_file
                research_request.files.append(uploaded_file)

        # Start background processing
        background_tasks.add_task(process_research_request, request_id)

        return research_request

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create research request: {str(e)}")
@router.get("/research/requests", response_model=List[ResearchRequest])
async def get_research_requests(
    status: Optional[ResearchStatus] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get all research requests with optional filtering"""
    try:
        requests_list = list(research_requests.values())
        
        if status:
            requests_list = [req for req in requests_list if req.status == status]
        
        # Sort by creation date (newest first)
        requests_list.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        paginated_requests = requests_list[offset:offset + limit]
        
        return paginated_requests
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch research requests: {str(e)}")

@router.get("/research/requests/{request_id}", response_model=ResearchRequest)
async def get_research_request(request_id: str):
    """Get a specific research request by ID"""
    if request_id not in research_requests:
        raise HTTPException(status_code=404, detail="Research request not found")
    
    return research_requests[request_id]

@router.put("/research/requests/{request_id}", response_model=ResearchRequest)
async def update_research_request(
    request_id: str,
    update: ResearchRequestUpdate
):
    """Update a research request"""
    if request_id not in research_requests:
        raise HTTPException(status_code=404, detail="Research request not found")
    
    request = research_requests[request_id]
    
    if update.status:
        request.status = update.status
        if update.status == ResearchStatus.completed:
            request.completed_at = datetime.now()
    
    if update.analysis:
        request.analysis = update.analysis
    
    if update.error:
        request.error = update.error
        request.status = ResearchStatus.error
    
    research_requests[request_id] = request
    return request

@router.delete("/research/requests/{request_id}")
async def delete_research_request(request_id: str):
    """Delete a research request"""
    if request_id not in research_requests:
        raise HTTPException(status_code=404, detail="Research request not found")
    
    # Clean up associated files
    request = research_requests[request_id]
    for file in request.files:
        if file.file_path and os.path.exists(file.file_path):
            os.remove(file.file_path)
        if file.id in uploaded_files:
            del uploaded_files[file.id]
    
    del research_requests[request_id]
    return {"message": "Research request deleted successfully"}

@router.post("/research/requests/{request_id}/files", response_model=UploadedFile)
async def upload_file_to_request(
    request_id: str,
    file: UploadFile = File(...)
):
    """Upload a file and associate it with a research request"""
    if request_id not in research_requests:
        raise HTTPException(status_code=404, detail="Research request not found")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="File type not supported")
    
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Read content for text files
        file_content = None
        if file_extension == '.txt':
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                file_content = await f.read()
        
        # Create file record
        uploaded_file = UploadedFile(
            id=file_id,
            name=file.filename,
            type=file.content_type or 'application/octet-stream',
            size=len(content),
            content=file_content,
            file_path=str(file_path)
        )
        
        # Store file record
        uploaded_files[file_id] = uploaded_file
        
        # Associate with research request
        research_requests[request_id].files.append(uploaded_file)
        
        return uploaded_file
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@router.get("/research/requests/{request_id}/analysis/export")
async def export_research_analysis(request_id: str):
    """Export research analysis as JSON"""
    if request_id not in research_requests:
        raise HTTPException(status_code=404, detail="Research request not found")
    
    request = research_requests[request_id]
    if not request.analysis:
        raise HTTPException(status_code=400, detail="No analysis available for export")
    
    export_data = {
        "request": request.dict(),
        "analysis": request.analysis.dict(),
        "exported_at": datetime.now().isoformat()
    }
    
    return export_data

@router.get("/references/{reference_id}/content")
async def get_reference_content(reference_id: str):
    """Fetch full content for a legal reference from Kenya Law"""
    # This would integrate with your existing Kenya Law search functionality
    try:
        # Mock implementation - replace with actual Kenya Law API calls
        await asyncio.sleep(1)  # Simulate API call delay
        
        # You would call your existing Kenya Law search functions here
        # content = await fetch_case_details(reference_id)
        
        mock_content = {
            "content": "# Mock Legal Document Content\n\nThis would contain the full legal document content fetched from Kenya Law database.",
            "source_url": f"http://kenyalaw.org/caselaw/cases/view/{reference_id}",
            "last_updated": datetime.now().isoformat()
        }
        
        return mock_content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reference content: {str(e)}")

# Background task for processing research requests
async def process_research_request(request_id: str):
    """Background task to process research requests with AI analysis"""
    try:
        if request_id not in research_requests:
            return
        
        request = research_requests[request_id]
        
        # Step 1: Analyzing
        request.status = ResearchStatus.analyzing
        research_requests[request_id] = request
        await asyncio.sleep(2)  # Simulate processing time
        
        # Step 2: Researching
        request.status = ResearchStatus.researching
        research_requests[request_id] = request
        await asyncio.sleep(3)  # Simulate research time
        
        # Step 3: Generate analysis (integrate with your AI/LLM here)
        analysis = await generate_ai_legal_analysis(request)
        
        # Step 4: Complete
        request.status = ResearchStatus.completed
        request.completed_at = datetime.now()
        request.analysis = analysis
        research_requests[request_id] = request
        
    except Exception as e:
        # Handle errors
        if request_id in research_requests:
            research_requests[request_id].status = ResearchStatus.error
            research_requests[request_id].error = str(e)

from datetime import datetime


async def generate_ai_legal_analysis(request: ResearchRequest) -> ResearchAnalysis:
    """Generate real AI-powered legal analysis using GPT"""

    # Extract text from uploaded .txt files
    supporting_texts = []
    for f in request.files:
        if f.content:  # already read .txt content
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

        # 🔧 Remove ```json ... ``` wrappers if present
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.DOTALL)

        # Try parse JSON
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("AI returned non-JSON response: %s", raw_content[:500])
            return create_fallback_analysis_object(request, "Invalid JSON from AI")

        # Normalize `type` values to lowercase for ReferenceType enum
        def create_legal_reference(item_data: dict) -> LegalReference:
            ref_type = str(item_data.get("type", "case")).lower()
            
            # Safely parse relevance
            raw_rel = item_data.get("relevance", 50)
            try:
                relevance = int(raw_rel)
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


def create_fallback_analysis(request: ResearchRequest, ai_content: str) -> dict:
    """Create fallback analysis when GPT doesn't return valid JSON"""
    return {
        "summary": f"AI analysis for {request.title}. Note: Response format required manual processing.",
        "key_findings": [
            "Analysis completed using AI assistance",
            "Manual review recommended for complex legal issues",
            "Consider consulting with legal experts"
        ],
        "legal_position": "Legal position requires further analysis based on specific case facts.",
        "recommendations": [
            "Review relevant Kenyan statutes and regulations",
            "Consider recent case law developments",
            "Seek expert legal opinion for complex matters"
        ],
        "risks": [
            "Potential compliance gaps",
            "Regulatory changes may affect legal position"
        ],
        "precedents": [],
        "statutes": [],
        "regulations": [],
        "related_cases": [],
        "full_analysis": f"# AI Legal Analysis\n\n{ai_content}"
    }
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
async def generate_legal_analysis(request: ResearchRequest) -> ResearchAnalysis:
    """Generate AI-powered legal analysis (integrate with your AI model here)"""
    
    # This is where you would integrate with your AI/LLM
    # For now, returning mock data similar to your frontend
    
    analysis_id = str(uuid.uuid4())
    
    # Mock legal references (you would get these from actual research)
    precedents = [
        LegalReference(
            id=str(uuid.uuid4()),
            type=ReferenceType.case,
            title="Republic v. Kenya National Highways Authority Ex Parte Citizens Coalition",
            citation="[2024] eKLR",
            relevance=95,
            summary="Landmark case establishing procedural requirements for public participation in infrastructure projects.",
            kenyan_law_url="http://kenyalaw.org/caselaw/cases/view/12345",
            year="2024"
        )
    ]
    
    statutes = [
        LegalReference(
            id=str(uuid.uuid4()),
            type=ReferenceType.statute,
            title="The Constitution of Kenya",
            citation="2010",
            relevance=100,
            summary="Supreme law of Kenya providing the fundamental legal framework.",
            kenyan_law_url="http://kenyalaw.org/kl/index.php?id=398",
            section="Articles 40, 47, 232"
        )
    ]
    
    analysis = ResearchAnalysis(
        id=analysis_id,
        summary=f"Comprehensive legal analysis for {request.title} has been completed. The research covers all relevant Kenyan laws, precedent cases, and regulatory frameworks.",
        key_findings=[
            "The Constitution of Kenya 2010 provides the fundamental framework for this matter",
            "Recent Court of Appeal decisions have clarified the legal position significantly",
            "New regulations published in 2024 impact the compliance requirements",
            "Cross-reference with East African Community directives is necessary"
        ],
        legal_position="Based on current Kenyan jurisprudence and statutory interpretation, the legal position is well-established with recent clarifications from the superior courts.",
        recommendations=[
            "Ensure compliance with the latest regulatory requirements",
            "Consider precedent set by recent High Court decisions",
            "Review constitutional implications under Articles 40 and 47",
            "Implement risk mitigation strategies for identified areas"
        ],
        risks=[
            "Potential constitutional challenges under the Bill of Rights",
            "Regulatory compliance gaps in new 2024 requirements",
            "Precedent uncertainty in emerging case law areas",
            "Cross-jurisdictional enforcement challenges"
        ],
        precedents=precedents,
        statutes=statutes,
        regulations=[],
        related_cases=[],
        full_analysis=f"""# Comprehensive Legal Analysis

## Executive Summary
This analysis examines {request.title} within the context of Kenyan law, considering constitutional provisions, statutory requirements, and recent jurisprudential developments.

## Legal Framework
The primary legal framework is established by the Constitution of Kenya 2010, specifically Articles 40 and 47, which guarantee fair administrative action and access to information respectively.

## Case Law Analysis
Recent decisions from the Court of Appeal and High Court have clarified several key aspects of this area of law, particularly regarding procedural requirements and constitutional compliance.

## Regulatory Compliance
New regulations published in 2024 have introduced additional compliance requirements that must be carefully considered in any implementation strategy.

## Risk Assessment
Several risk factors have been identified, including potential constitutional challenges and regulatory compliance gaps that require immediate attention.

## Recommendations
Based on this comprehensive analysis, specific recommendations have been developed to ensure legal compliance and minimize identified risks."""
    )
    
    return analysis

# Statistics endpoint
@router.get("/research/statistics")
async def get_research_statistics():
    """Get research statistics"""
    total_requests = len(research_requests)
    completed_requests = len([r for r in research_requests.values() if r.status == ResearchStatus.completed])
    pending_requests = len([r for r in research_requests.values() if r.status in [ResearchStatus.pending, ResearchStatus.analyzing, ResearchStatus.researching]])
    error_requests = len([r for r in research_requests.values() if r.status == ResearchStatus.error])
    
    # Request type breakdown
    type_breakdown = {}
    for request_type in RequestType:
        type_breakdown[request_type.value] = len([r for r in research_requests.values() if r.type == request_type])
    
    return {
        "total_requests": total_requests,
        "completed_requests": completed_requests,
        "pending_requests": pending_requests,
        "error_requests": error_requests,
        "completion_rate": (completed_requests / total_requests * 100) if total_requests > 0 else 0,
        "type_breakdown": type_breakdown
    }