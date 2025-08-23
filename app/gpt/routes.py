from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import openai
import os
from dotenv import load_dotenv
from datetime import datetime
import json
import logging

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/legal", tags=["kenya-legal"])

# Request schemas
class ChatRequest(BaseModel):
    message: str
    legal_area: Optional[str] = Field(default="general", description="Legal area of focus")
    case_preference: Optional[str] = Field(default="recent", description="Preference for case law (recent, landmark, all)")
    citation_level: Optional[str] = Field(default="medium", description="Citation detail level (low, medium, high)")
    context_history: Optional[List[Dict[str, str]]] = Field(default=[], description="Previous conversation context")

class LegalQuery(BaseModel):
    query: str
    search_type: str = Field(..., description="Type: case_law, statutes, regulations, contracts")
    jurisdiction: Optional[str] = Field(default="kenya", description="Legal jurisdiction")
    date_range: Optional[Dict[str, str]] = Field(default=None, description="Date range for cases")

# Enhanced legal contexts and system prompts
LEGAL_CONTEXTS = {
    "general": {
        "system_prompt": """You are KenyaLex, an expert Kenyan legal research assistant with comprehensive knowledge of:
        - Kenyan Constitution (2010)
        - Acts of Parliament and subsidiary legislation
        - High Court, Court of Appeal, and Supreme Court decisions
        - Commercial law, contract law, employment law, family law, criminal law
        - Legal procedures and court processes in Kenya
        
        Always:
        1. Cite relevant Kenyan statutes, cases, and legal provisions
        2. Reference specific sections of laws when applicable
        3. Distinguish between binding and persuasive precedents
        4. Provide practical legal guidance while noting limitations
        5. Include disclaimer that this is not formal legal advice
        6. Use clear, professional language suitable for both lawyers and laypersons""",
        "keywords": ["general legal", "kenyan law", "legal guidance"]
    },
    
    "constitutional": {
        "system_prompt": """You are a constitutional law specialist focused on the Constitution of Kenya 2010. 
        Expertise areas:
        - Bill of Rights (Chapter 4)
        - Devolution and county governments
        - Judicial independence and court structure
        - Constitutional interpretation principles
        - Fundamental rights and freedoms
        
        Reference specific articles, cite constitutional court cases, and explain constitutional principles clearly.""",
        "keywords": ["constitution", "bill of rights", "fundamental rights"]
    },
    
    "contract_law": {
        "system_prompt": """You are a Kenyan contract law expert specializing in:
        - Contract formation, terms, and interpretation under Kenyan law
        - Sale of Goods Act, Law of Contract Act
        - Commercial contracts, employment contracts, real estate
        - Breach of contract remedies and damages
        - Standard form contracts and unfair terms
        
        Always cite relevant case law from Kenyan courts and applicable statutory provisions.""",
        "keywords": ["contracts", "agreements", "breach", "commercial law"]
    },
    
    "employment_law": {
        "system_prompt": """You are a Kenyan employment law specialist with expertise in:
        - Employment Act 2007 and Labour Relations Act 2007
        - Work Injury Benefits Act 2007
        - Occupational Safety and Health Act 2007
        - Employment and Labour Relations Court decisions
        - Collective bargaining agreements and trade unions
        - Wrongful dismissal, discrimination, and workplace rights
        
        Provide practical advice on employment disputes and compliance.""",
        "keywords": ["employment", "labour", "workplace", "dismissal", "discrimination"]
    },
    
    "criminal_law": {
        "system_prompt": """You are a Kenyan criminal law expert specializing in:
        - Penal Code (Cap 63) and Criminal Procedure Code (Cap 75)
        - Evidence Act and criminal evidence rules
        - Bail, sentencing, and appeals
        - Constitutional rights of accused persons
        - High Court and Court of Appeal criminal decisions
        
        Focus on substantive criminal law, procedure, and recent judicial precedents.""",
        "keywords": ["criminal", "penal code", "evidence", "bail", "sentencing"]
    },
    
    "family_law": {
        "system_prompt": """You are a Kenyan family law specialist covering:
        - Marriage Act 2014 and matrimonial law
        - Children Act 2022 and child protection
        - Succession laws and inheritance disputes
        - Divorce, custody, and maintenance
        - Gender-based violence and protection orders
        
        Provide sensitive, practical guidance on family legal matters.""",
        "keywords": ["marriage", "divorce", "children", "custody", "succession"]
    },
    
    "commercial_law": {
        "system_prompt": """You are a Kenyan commercial law expert specializing in:
        - Companies Act 2015 and corporate governance
        - Partnership Act and business structures
        - Insolvency Act 2015 and bankruptcy
        - Banking laws and financial services regulation
        - Competition Act and consumer protection
        - Commercial Court decisions and business disputes
        
        Focus on practical business law guidance and regulatory compliance.""",
        "keywords": ["companies", "corporate", "business", "commercial disputes"]
    },
    
    "land_law": {
        "system_prompt": """You are a Kenyan land law specialist with expertise in:
        - Land Act 2012 and Land Registration Act 2012
        - National Land Commission Act
        - Community Land Act and land rights
        - Compulsory acquisition and compensation
        - Land disputes and Environment and Land Court decisions
        
        Provide guidance on land ownership, transfers, and disputes.""",
        "keywords": ["land", "property", "title", "acquisition", "land disputes"]
    }
}

# Legal citation patterns and templates
CITATION_TEMPLATES = {
    "case": "{case_name} [{year}] eKLR or [{year}] {court_level} {citation_number}",
    "statute": "{act_name}, {year}, Section {section}",
    "constitution": "Constitution of Kenya, 2010, Article {article}",
    "regulation": "{regulation_name}, {year}, Regulation {number}"
}

def get_legal_context(legal_area: str) -> Dict[str, Any]:
    """Get appropriate legal context based on area of law"""
    return LEGAL_CONTEXTS.get(legal_area.lower(), LEGAL_CONTEXTS["general"])

# Model configuration - set your preferred model here
PREFERRED_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Default to gpt-4o-mini

def get_available_model() -> str:
    """Get the configured OpenAI model"""
    return PREFERRED_MODEL

def enhance_legal_prompt(base_prompt: str, query: str, legal_area: str) -> str:
    """Enhance the system prompt with specific legal focus"""
    context = get_legal_context(legal_area)
    
    enhanced_prompt = f"""{context['system_prompt']}

CURRENT QUERY CONTEXT:
Legal Area: {legal_area}
Keywords to focus on: {', '.join(context['keywords'])}

RESPONSE REQUIREMENTS:
1. Start with a brief summary of the legal issue
2. Cite relevant Kenyan laws, cases, and precedents
3. Explain the legal principles involved
4. Provide practical guidance where appropriate
5. Include relevant procedural information if applicable
6. End with appropriate legal disclaimers

IMPORTANT: Always specify that this is general legal information and not formal legal advice. Recommend consulting with a qualified Kenyan lawyer for specific legal matters.
"""
    return enhanced_prompt

@router.post("/chat")
async def legal_chat(req: ChatRequest):
    """Main chat endpoint for legal queries"""
    try:
        # Get appropriate legal context
        system_prompt = enhance_legal_prompt("", req.message, req.legal_area)
        
        # Build conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add context history if provided
        if req.context_history:
            for msg in req.context_history[-5:]:  # Keep last 5 exchanges
                messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": req.message})
        
        # Call OpenAI API with model fallback
        model = get_available_model()
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,  # Increased for comprehensive legal responses
            temperature=0.3,  # Lower temperature for more accurate legal information
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        reply = response.choices[0].message.content
        
        # Log the interaction (without sensitive data)
        logger.info(f"Legal query processed - Area: {req.legal_area}, Query length: {len(req.message)}")
        
        return {
            "reply": reply,
            "legal_area": req.legal_area,
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "This response provides general legal information only and does not constitute formal legal advice. Please consult with a qualified Kenyan lawyer for specific legal matters."
        }
        
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/research")
async def legal_research(req: LegalQuery):
    """Specialized endpoint for focused legal research"""
    try:
        research_prompts = {
            "case_law": f"""You are a Kenyan legal researcher specializing in case law analysis.
            Research the following query focusing on:
            - Relevant High Court, Court of Appeal, and Supreme Court decisions
            - Binding vs. persuasive precedents
            - Recent developments in jurisprudence
            - Case citations in proper format
            
            Query: {req.query}""",
            
            "statutes": f"""You are a Kenyan legislative research specialist.
            Research the following focusing on:
            - Relevant Acts of Parliament and their provisions
            - Subsidiary legislation and regulations
            - Recent amendments and their implications
            - Cross-references between related laws
            
            Query: {req.query}""",
            
            "regulations": f"""You are a Kenyan regulatory compliance specialist.
            Research the following focusing on:
            - Relevant regulations and statutory instruments
            - Licensing and compliance requirements
            - Regulatory body guidelines
            - Practical compliance steps
            
            Query: {req.query}""",
            
            "contracts": f"""You are a Kenyan contract law specialist.
            Research the following focusing on:
            - Standard contract clauses and terms
            - Legal requirements for validity
            - Common pitfalls and how to avoid them
            - Template language and best practices
            
            Query: {req.query}"""
        }
        
        system_prompt = research_prompts.get(req.search_type, research_prompts["case_law"])
        
        response = openai.chat.completions.create(
            model=get_available_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.query}
            ],
            max_tokens=1200,
            temperature=0.2  # Very low temperature for research accuracy
        )
        
        return {
            "research_result": response.choices[0].message.content,
            "search_type": req.search_type,
            "jurisdiction": req.jurisdiction,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Research error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Research error: {str(e)}")

@router.get("/legal-areas")
async def get_legal_areas():
    """Get available legal specialization areas"""
    return {
        "legal_areas": list(LEGAL_CONTEXTS.keys()),
        "descriptions": {area: context["keywords"] for area, context in LEGAL_CONTEXTS.items()}
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Kenya Legal Research Assistant"}
    raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found"
        )