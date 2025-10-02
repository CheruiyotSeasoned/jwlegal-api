from fastapi import APIRouter, HTTPException, Query, Depends,BackgroundTasks
from typing import List, Dict, Optional, Any
import requests
import time
import json
import os
from dotenv import load_dotenv
import re
import hashlib
import uuid
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import asyncio
import logging
import openai
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from openai import OpenAI
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken
from sentence_transformers import SentenceTransformer
qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333"))
)
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

router = APIRouter(prefix="/kenyalaw", tags=["Kenya Law"])
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class DocumentRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID to process")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Legal question")
    limit: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    min_score: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity score")

class Source(BaseModel):
    doc_id: str
    title: str
    court: Optional[str] = None
    date: Optional[str] = None
    citations: Optional[List[str]] = None
    snippet: str
    source_html_url: Optional[str] = None
    score: float

class RAGResponse(BaseModel):
    answer: str
    sources: List[Source]
    query: str
    processing_time: float
    tokens_used: Optional[int] = None

# Configuration
class RAGConfig:
    COLLECTION_NAME = "kenya_cases"
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 80
    MAX_CONTEXT_TOKENS = 8000
    CHAT_MODEL = "gpt-4o-mini"  # Only for chat completion
    MAX_RETRIES = 3
    RATE_LIMIT_DELAY = 1.0
    
    # Embedding model options
    EMBEDDING_MODELS = {
        "bge-large": {"name": "BAAI/bge-large-en-v1.5", "size": 1024},
        "bge-base": {"name": "BAAI/bge-base-en-v1.5", "size": 768},
        "bge-small": {"name": "BAAI/bge-small-en-v1.5", "size": 384},
        "all-mpnet": {"name": "all-mpnet-base-v2", "size": 768},
        "all-miniLM": {"name": "all-MiniLM-L6-v2", "size": 384},
        "nomic": {"name": "nomic-ai/nomic-embed-text-v1", "size": 768},
        "e5-large": {"name": "intfloat/e5-large-v2", "size": 1024},
        "e5-base": {"name": "intfloat/e5-base-v2", "size": 768},
    }
    
    # Choose your embedding model here
    SELECTED_MODEL = "bge-base"  # Good balance of quality and speed
    
    @property
    def VECTOR_SIZE(self):
        return self.EMBEDDING_MODELS[self.SELECTED_MODEL]["size"]
    
    @property
    def EMBEDDING_MODEL_NAME(self):
        return self.EMBEDDING_MODELS[self.SELECTED_MODEL]["name"]

config = RAGConfig()

# Initialize embedding model
logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
logger.info(f"Embedding model loaded. Vector size: {config.VECTOR_SIZE}")

# Pydantic models for request/response
class SearchRequest(BaseModel):
    search_term: str
    page: int = 1
    page_size: int = 20
    court_filter: Optional[str] = None
    year_filter: Optional[str] = None
    use_cache: bool = True
    cache_max_age_hours: int = 24
    
    # Enhanced AI options
    ai_enhancement_level: str = "traditional"  # "none", "traditional", "gpt", "hybrid"
    gpt_rerank_top_n: int = 10  # Only use GPT for top N results
    fetch_full_content: bool = False  # More conservative default
    openai_api_key: Optional[str] = None  # For GPT features

class SearchResponse(BaseModel):
    results: List[Dict]
    total_count: int
    total_pages: int
    current_page: int
    facets: Optional[Dict] = None
    cached_at: str
    search_params: Dict
    from_cache: bool = False
    ai_enhanced: bool = False
    enhancement_stats: Optional[Dict] = None  # New field for AI stats

class CacheStats(BaseModel):
    total_searches: int
    total_documents: int
    cache_size_mb: float
    created_at: Optional[str]
    last_updated: Optional[str]
def get_cache_filepath() -> str:
    """Get the filepath for the main cache file."""
    cache_dir = "kenyalaw_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return os.path.join(cache_dir, "kenyalaw_cache.json")

def get_search_key(search_term: str, page: int, page_size: int, court_filter: Optional[str] = None, 
                  year_filter: Optional[str] = None, ai_enhanced: bool = False, fetch_full: bool = False, 
                  enhancement_level: str = "traditional") -> str:
    """Generate a unique search key for the search parameters."""
    cache_string = f"{search_term}_{page}_{page_size}_{court_filter}_{year_filter}_{ai_enhanced}_{fetch_full}_{enhancement_level}"
    return hashlib.md5(cache_string.encode()).hexdigest()

def load_cache() -> Dict:
    """Load the main cache file."""
    cache_filepath = get_cache_filepath()
    try:
        if os.path.exists(cache_filepath):
            with open(cache_filepath, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                return cache_data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    return {
        "searches": {},
        "documents": {},
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_searches": 0,
            "total_documents": 0
        }
    }

def save_cache(cache_data: Dict):
    """Save the main cache file."""
    cache_filepath = get_cache_filepath()
    try:
        cache_data["metadata"]["last_updated"] = datetime.now().isoformat()
        cache_data["metadata"]["total_searches"] = len(cache_data.get("searches", {}))
        cache_data["metadata"]["total_documents"] = len(cache_data.get("documents", {}))
        
        with open(cache_filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

def is_cache_valid(cache_data: Dict, max_age_hours: int = 24) -> bool:
    """Check if cache data is still valid."""
    last_updated = cache_data.get("metadata", {}).get("last_updated")
    if not last_updated:
        return False
    
    try:
        last_update_time = datetime.fromisoformat(last_updated)
        return datetime.now() - last_update_time < timedelta(hours=max_age_hours)
    except:
        return False

def get_cached_search(search_key: str, cache_data: Dict) -> Optional[Dict]:
    """Get cached search results."""
    return cache_data.get("searches", {}).get(search_key)

def cache_search_results(search_key: str, search_data: Dict[str, Any], cache_data: Dict[str, Any]):
    """Cache search results and individual documents safely."""
    cache_data.setdefault("searches", {})
    cache_data.setdefault("documents", {})

    # Always cache the raw search data
    cache_data["searches"][search_key] = {
        "data": search_data,
        "cached_at": datetime.now().isoformat(),
        "search_params": search_data.get("search_params", {})
    }

    documents = cache_data["documents"]

    # --- Normalize results ---
    results = search_data.get("results", [])
    # Handle nested [[]]
    if results and isinstance(results[0], list):
        results = results[0]

    # --- Extract individual docs ---
    for doc in results:
        if not isinstance(doc, dict):
            continue

        doc_id = doc.get("id") or doc.get("document_id") or doc.get("doc_id")
        if not doc_id:
            continue

        doc_id_str = str(doc_id)

        if doc_id_str not in documents:
            documents[doc_id_str] = {
                "data": doc,
                "cached_at": datetime.now().isoformat(),
                "source_searches": [search_key]
            }
        else:
            if search_key not in documents[doc_id_str]["source_searches"]:
                documents[doc_id_str]["source_searches"].append(search_key)

    cache_data["documents"] = documents

async def fetch_document_html_content(doc_id: str) -> Optional[str]:
    """Fetch full HTML content for a document."""
    try:
        cache_data = load_cache()
        cached_doc = cache_data.get("documents", {}).get(str(doc_id), {}).get("data")
        
        if not cached_doc:
            url = f"https://new.kenyalaw.org/search/api/documents/{doc_id}/"
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return None
            cached_doc = response.json()
        
        expression_uri = cached_doc.get("expression_frbr_uri") or cached_doc.get("work_frbr_uri")
        if not expression_uri:
            search_url = f"https://new.kenyalaw.org/search/api/documents/?q=id:{doc_id}"
            sr = requests.get(search_url, timeout=30)
            if sr.status_code == 200:
                sjson = sr.json()
                results = sjson.get("results", [])
                if results:
                    expression_uri = results[0].get("expression_frbr_uri") or results[0].get("work_frbr_uri")
        
        if not expression_uri:
            return None
            
        html_url = f"https://new.kenyalaw.org{expression_uri}"
        page_res = requests.get(html_url, timeout=30)
        if page_res.status_code != 200:
            return None
            
        page_html = page_res.text
        soup = BeautifulSoup(page_html, "html.parser")
        
        for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
            tag.decompose()
        
        content_el = (
            soup.select_one("#document_content") or
            soup.select_one("div.content__html") or
            soup.select_one("div.document-content__inner") or
            soup.select_one("div.document-content") or
            soup.select_one("div.content-and-enrichments__inner") or
            soup.body
        )
        
        if content_el:
            return content_el.get_text(separator="\n", strip=True)
        
        return None
        
    except Exception as e:
        logger.warning(f"Error fetching HTML content for doc {doc_id}: {e}")
        return None


@asynccontextmanager
async def lifespan(app):
    """Application lifespan events."""
    # Startup
    await init_collection()
    yield
    # Shutdown
    logger.info("Shutting down RAG service")

async def init_collection():
    """Initialize Qdrant collection with proper error handling."""
    try:
        # Check if collection exists
        collections = qdrant.get_collections()
        collection_exists = any(
            col.name == config.COLLECTION_NAME 
            for col in collections.collections
        )
        
        if collection_exists:
            # Check if dimensions match
            try:
                collection_info = qdrant.get_collection(config.COLLECTION_NAME)
                existing_size = collection_info.config.params.vectors.size
                
                if existing_size != config.VECTOR_SIZE:
                    logger.warning(f"Collection exists with wrong vector size: {existing_size}, expected: {config.VECTOR_SIZE}")
                    logger.info("Recreating collection with correct dimensions...")
                    
                    # Delete existing collection
                    qdrant.delete_collection(config.COLLECTION_NAME)
                    collection_exists = False
                else:
                    logger.info(f"Collection {config.COLLECTION_NAME} exists with correct dimensions")
                    
            except Exception as e:
                logger.error(f"Error checking collection dimensions: {e}")
                # Delete and recreate to be safe
                qdrant.delete_collection(config.COLLECTION_NAME)
                collection_exists = False
        
        if not collection_exists:
            logger.info(f"Creating collection: {config.COLLECTION_NAME} with {config.VECTOR_SIZE} dimensions")
            qdrant.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=config.VECTOR_SIZE, 
                    distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfig(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                ),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            )
            logger.info("Collection created successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize collection: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Database initialization failed: {str(e)}"
        )

def chunk_text(text: str, chunk_words: int = None, overlap: int = None) -> List[str]:
    """Split text into overlapping chunks."""
    chunk_words = chunk_words or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    
    if not text or not text.strip():
        return []
    
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    
    chunks = []
    for i in range(0, len(words), chunk_words - overlap):
        chunk_words_slice = words[i:i + chunk_words]
        if not chunk_words_slice:
            break
            
        chunk = " ".join(chunk_words_slice)
        
        if len(chunk_words_slice) >= overlap or i + chunk_words >= len(words):
            chunks.append(chunk)
            
        if i + chunk_words >= len(words):
            break
    
    return chunks

async def embed_text(text: str) -> List[float]:
    """Generate embeddings using local model."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    try:
        # Run embedding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            lambda: embedding_model.encode(text.strip(), convert_to_tensor=False)
        )
        
        # Convert numpy array to list
        embedding_list = embedding.tolist()
        
        if len(embedding_list) != config.VECTOR_SIZE:
            raise ValueError(f"Unexpected embedding size: {len(embedding_list)}")
            
        return embedding_list
        
    except Exception as e:
        logger.error(f"Local embedding failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Embedding generation failed: {str(e)}"
        )

async def upsert_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Index Kenya Law case into Qdrant."""
    start_time = datetime.now()
    
    # Validate document
    if not doc:
        raise ValueError("Document is empty")
    
    content = doc.get("content_text", "").strip()
    if not content:
        raise ValueError("Document content is empty")
    
    case_number = doc.get("case_number") or doc.get("doc_id")
    if not case_number:
        raise ValueError("Document must have case_number or doc_id")
    
    try:
        # Generate chunks
        chunks = chunk_text(content)
        if not chunks:
            raise ValueError("No chunks generated from document")
        
        logger.info(f"Processing document {case_number}: {len(chunks)} chunks")
        
        # Generate embeddings for all chunks
        points = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = await embed_text(chunk)
                
                # Generate UUID from deterministic hash
                hash_input = f"{case_number}_{i}_{chunk[:50]}".encode()
                hash_bytes = hashlib.md5(hash_input).digest()
                chunk_uuid = str(uuid.UUID(bytes=hash_bytes))
                
                points.append(models.PointStruct(
                    id=chunk_uuid,
                    vector=embedding,
                    payload={
                        "doc_id": case_number,
                        "chunk_index": i,
                        "title": doc.get("title", "Unknown Title"),
                        "date": doc.get("date"),
                        "court": doc.get("court"),
                        "citations": doc.get("citations", []),
                        "snippet": chunk[:500],
                        "source_html_url": doc.get("source_html_url"),
                        "indexed_at": datetime.utcnow().isoformat(),
                        "chunk_length": len(chunk),
                    }
                ))
                
            except Exception as e:
                logger.error(f"Failed to process chunk {i}: {str(e)}")
                continue
        
        if not points:
            raise ValueError("No chunks were successfully processed")
        
        # Upsert to Qdrant
        try:
            operation_result = qdrant.upsert(
                collection_name=config.COLLECTION_NAME, 
                points=points,
                wait=True
            )
            
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store document vectors: {str(e)}"
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            "doc_id": case_number,
            "chunks_processed": len(points),
            "processing_time": processing_time,
            "indexed_at": datetime.utcnow().isoformat(),
            "embedding_model": config.EMBEDDING_MODEL_NAME
        }
        
        logger.info(f"Document {case_number} indexed successfully: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Document indexing failed for {case_number}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Document indexing failed: {str(e)}"
        )

async def index_document(request: DocumentRequest):
    """Index a document by ID."""
    try:
        # You need to implement get_document_html
        doc = await get_document_html(request.doc_id)
        result = await upsert_document(doc)
        logger.info(f"Document indexing endpoint completed for {request.doc_id}")
        logger.info(f"Indexing result: {doc.get('title')} | {result}")
        logger.info(f"Document indexed: {doc}")
        
        return {
            **result,
            "status": "indexed",
            "document": {
                "title": doc.get("title"),
                "case_number": doc.get("case_number"),
                "court": doc.get("court"),
                "date": doc.get("date")
            }
        }
        
    except Exception as e:
        logger.error(f"Document indexing endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_document(document_id: str):
    """Get a specific document by ID from cache or API."""
    try:
        # First check cache
        cache_data = load_cache()
        cached_doc = cache_data.get("documents", {}).get(str(document_id), {}).get("data")
        
        if cached_doc:
            return {
                "document": cached_doc,
                "from_cache": True,
                "source_searches": cache_data.get("documents", {}).get(str(document_id), {}).get("source_searches", [])
            }
        
        # If not in cache, try to fetch from API
        url = f"https://new.kenyalaw.org/search/api/documents/{document_id}/"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            return {
                "document": response.json(),
                "from_cache": False,
                "source_searches": []
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

async def get_document_html(doc_id: str):
    """Get full HTML content and metadata for a document."""
    try:
        # Re-use the get_document method to get metadata
        meta_resp = await get_document(doc_id)
        metadata = None
        if isinstance(meta_resp, dict):
            metadata = meta_resp.get("document") or meta_resp
        else:
            metadata = meta_resp

        if not metadata:
            raise HTTPException(status_code=404, detail="Document metadata not found")

        # Resolve expression_frbr_uri
        expression_uri = metadata.get("expression_frbr_uri") or metadata.get("work_frbr_uri")
        if not expression_uri:
            # fallback: query search API for id
            search_url = f"https://new.kenyalaw.org/search/api/documents/?q=id:{doc_id}"
            sr = requests.get(search_url, timeout=30)
            sr.raise_for_status()
            sjson = sr.json()
            results = sjson.get("results", [])
            if not results:
                raise HTTPException(status_code=404, detail="Document expression URI not found")
            expression_uri = results[0].get("expression_frbr_uri") or results[0].get("work_frbr_uri")

        if not expression_uri:
            raise HTTPException(status_code=404, detail="No expression URI available")

        # Fetch the HTML page
        html_url = f"https://new.kenyalaw.org{expression_uri}"
        page_res = requests.get(html_url, timeout=30)
        page_res.raise_for_status()
        page_html = page_res.text

        # Parse HTML
        soup = BeautifulSoup(page_html, "html.parser")

        # Remove unwanted elements
        for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
            tag.decompose()

        # Get main content
        content_el = (
            soup.select_one("#document_content") or
            soup.select_one("div.content__html") or
            soup.select_one("div.document-content__inner") or
            soup.select_one("div.document-content") or
            soup.select_one("div.content-and-enrichments__inner") or
            soup.body
        )

        content_html = content_el.decode_contents() if content_el else ""
        content_text = content_el.get_text(separator="\n", strip=True) if content_el else soup.get_text("\n", strip=True)

        # Extract metadata from HTML
        html_metadata = {}
        for dl in soup.select("dl.document-metadata-list"):
            dts = dl.find_all("dt")
            for dt in dts:
                dd = dt.find_next_sibling("dd")
                if dd:
                    key = dt.get_text(strip=True)
                    val = " ".join(dd.stripped_strings)
                    html_metadata[key.lower().replace(" ", "_")] = val

        merged_meta = { **(metadata or {}), **html_metadata }

        # Extract attachments
        attachments = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if "kenyalaw-website-media" in href or "/attachments/" in href:
                attachments.append({
                    "text": a.get_text(strip=True),
                    "url": href
                })
        seen = set()
        attachments = [a for a in attachments if (a["url"] not in seen and not seen.add(a["url"]))]

        # Extract citations
        citations = []
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if "/akn/ke/judgment/" in href or "/judgments/" in href or "KEHC" in a.get_text():
                txt = a.get_text(strip=True)
                if txt:
                    citations.append(txt)
        bracket_cites = re.findall(r"\[\d{4}\]\s*[A-Z]{2,}[^\n\(\)]+(?:\([A-Z]{3,}\))?", content_text)
        citations = list(dict.fromkeys(citations + bracket_cites))

        # Extract sections
        sections = {}
        headings = re.split(r"\n([A-Z][A-Z\s]{3,})\n", content_text)
        if len(headings) > 1:
            pre = headings[0].strip()
            if pre:
                sections["preface"] = pre
            for i in range(1, len(headings), 2):
                h = headings[i].strip().lower()
                body = headings[i+1].strip() if i+1 < len(headings) else ""
                sections[h] = body
        return {
            "title": merged_meta.get("title") or merged_meta.get("citation") or soup.title.string if soup.title else None,
            "date": merged_meta.get("judgment_date") or merged_meta.get("judgment date") or merged_meta.get("date") or metadata.get("date"),
            "court": merged_meta.get("court") or metadata.get("court"),
            "registry": merged_meta.get("court_station") or merged_meta.get("registry") or metadata.get("registry"),
            "case_number": merged_meta.get("case_number") or metadata.get("case_number"),
            "judges": metadata.get("judges") or merged_meta.get("judges"),
            "attorneys": merged_meta.get("attorneys"),
            "attachments": attachments,
            "citations": citations,
            "sections": sections,
            "word_count": len(content_text.split()),
            "content_text": content_text,
            "content_html": content_html,
            "source_html_url": html_url,
            "from_cache": True if isinstance(meta_resp, dict) and meta_resp.get("from_cache") else False
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")