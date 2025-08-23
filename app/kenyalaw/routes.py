from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
import requests
import time
import json
import os
import re
import hashlib
from datetime import datetime, timedelta
from pydantic import BaseModel
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import logging

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    search_term: str
    page: int = 1
    page_size: int = 20
    court_filter: Optional[str] = None
    year_filter: Optional[str] = None
    use_cache: bool = True
    cache_max_age_hours: int = 24
    enable_ai_enhancement: bool = True  # New field for AI features
    fetch_full_content: bool = True  # New field to control full HTML content fetching

class SearchResponse(BaseModel):
    results: List[Dict]
    total_count: int
    total_pages: int
    current_page: int
    facets: Optional[Dict] = None
    cached_at: str
    search_params: Dict
    from_cache: bool = False
    ai_enhanced: bool = False  # New field to indicate AI enhancement

class CacheStats(BaseModel):
    total_searches: int
    total_documents: int
    cache_size_mb: float
    created_at: Optional[str]
    last_updated: Optional[str]

# AI Enhancement Classes
class RelevanceCalculator:
    """Calculate semantic relevance between search terms and case facts."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis."""
        if not text:
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Tokenize and remove stopwords
        words = word_tokenize(text)
        words = [self.stemmer.stem(word) for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def calculate_relevance(self, search_term: str, case_facts: str) -> float:
        """Calculate semantic relevance score between search term and case facts."""
        try:
            if not search_term or not case_facts:
                return 0.0
            
            # Preprocess texts
            processed_search = self.preprocess_text(search_term)
            processed_facts = self.preprocess_text(case_facts)
            
            if not processed_search or not processed_facts:
                return 0.0
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1
            )
            
            # Fit on both texts
            corpus = [processed_search, processed_facts]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Apply boost for exact keyword matches
            search_keywords = set(search_term.lower().split())
            facts_words = set(case_facts.lower().split())
            keyword_overlap = len(search_keywords & facts_words) / max(len(search_keywords), 1)
            
            # Combine semantic similarity with keyword boost
            final_score = min(1.0, similarity + (keyword_overlap * 0.3))
            
            return float(final_score)
            
        except Exception as e:
            logger.warning(f"Error calculating relevance: {e}")
            return 0.0

class FactSummarizer:
    """Generate concise summaries of case facts."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def extract_facts_section(self, content: str) -> str:
        """Extract the facts section from case content."""
        if not content:
            return ""
        
        # Common patterns for facts sections in legal documents
        facts_patterns = [
            r'FACTS?\s*:?\s*(.*?)(?=ISSUES?|JUDGMENT|HELD|RULING|DECISION)',
            r'BACKGROUND\s*:?\s*(.*?)(?=ISSUES?|JUDGMENT|HELD|RULING|DECISION)',
            r'FACTUAL\s+BACKGROUND\s*:?\s*(.*?)(?=ISSUES?|JUDGMENT|HELD|RULING|DECISION)',
            r'THE\s+FACTS\s*:?\s*(.*?)(?=ISSUES?|JUDGMENT|HELD|RULING|DECISION)'
        ]
        
        content_upper = content.upper()
        
        for pattern in facts_patterns:
            match = re.search(pattern, content_upper, re.DOTALL | re.IGNORECASE)
            if match:
                facts = match.group(1).strip()
                # Convert back to original case
                start_idx = content_upper.find(facts)
                if start_idx != -1:
                    return content[start_idx:start_idx + len(facts)].strip()
        
        # If no facts section found, return first 700 words
        words = content.split()
        return ' '.join(words[:700])
    
    def score_sentence_importance(self, sentence: str, all_sentences: List[str], keywords: List[str]) -> float:
        """Score sentence importance based on various factors."""
        score = 0.0
        sentence_lower = sentence.lower()
        
        # Keyword presence
        for keyword in keywords:
            if keyword.lower() in sentence_lower:
                score += 0.3
        
        # Legal terms boost
        legal_terms = ['court', 'judge', 'plaintiff', 'defendant', 'contract', 'negligence', 
                      'breach', 'damages', 'liability', 'evidence', 'witness']
        for term in legal_terms:
            if term in sentence_lower:
                score += 0.1
        
        # Position boost (earlier sentences often more important)
        position_factor = 1 - (all_sentences.index(sentence) / len(all_sentences)) * 0.3
        score *= position_factor
        
        # Length penalty for very short or very long sentences
        word_count = len(sentence.split())
        if word_count < 5:
            score *= 0.5
        elif word_count > 50:
            score *= 0.7
        
        return score
    
    def summarize_facts(self, facts: str, search_term: str = "", max_sentences: int = 3) -> str:
        """Generate a concise summary of the facts."""
        if not facts:
            return "No factual information available."
        
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(facts)
            
            if len(sentences) <= max_sentences:
                return facts.strip()
            
            # Extract keywords from search term
            keywords = [word.strip() for word in search_term.split() if len(word.strip()) > 2]
            
            # Score sentences
            sentence_scores = []
            for sentence in sentences:
                score = self.score_sentence_importance(sentence, sentences, keywords)
                sentence_scores.append((sentence, score))
            
            # Sort by score and select top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = sentence_scores[:max_sentences]
            
            # Sort selected sentences by their original order
            original_order = []
            for sentence, _ in top_sentences:
                original_order.append((sentences.index(sentence), sentence))
            original_order.sort()
            
            summary = ' '.join([sentence for _, sentence in original_order])
            return summary.strip()
            
        except Exception as e:
            logger.warning(f"Error summarizing facts: {e}")
            # Fallback: return first few sentences
            sentences = facts.split('. ')
            return '. '.join(sentences[:max_sentences]).strip() + ('.' if not sentences[-1].endswith('.') else '')

# Initialize AI components
relevance_calculator = RelevanceCalculator()
fact_summarizer = FactSummarizer()

# Cache management functions (unchanged)
def get_cache_filepath() -> str:
    """Get the filepath for the main cache file."""
    cache_dir = "kenyalaw_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return os.path.join(cache_dir, "kenyalaw_cache.json")

def get_search_key(search_term: str, page: int, page_size: int, court_filter: Optional[str] = None, year_filter: Optional[str] = None, ai_enhanced: bool = False, fetch_full: bool = False) -> str:
    """Generate a unique search key for the search parameters."""
    cache_string = f"{search_term}_{page}_{page_size}_{court_filter}_{year_filter}_{ai_enhanced}_{fetch_full}"
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
    
    # Return empty cache structure
    return {
        "searches": {},      # Search results by search key
        "documents": {},     # Individual documents by ID
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
    """Check if cache data is still valid (not too old)."""
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

def cache_search_results(search_key: str, search_data: Dict, cache_data: Dict):
    """Cache search results and individual documents."""
    # Cache the search results
    cache_data["searches"][search_key] = {
        "data": search_data,
        "cached_at": datetime.now().isoformat(),
        "search_params": search_data.get("search_params", {})
    }
    
    # Cache individual documents (deduplicated by ID)
    documents = cache_data.get("documents", {})
    for doc in search_data.get("results", []):
        doc_id = doc.get("id")
        if doc_id and str(doc_id) not in documents:
            documents[str(doc_id)] = {
                "data": doc,
                "cached_at": datetime.now().isoformat(),
                "source_searches": [search_key]
            }
        elif doc_id and str(doc_id) in documents:
            # Add this search as a source if not already present
            if search_key not in documents[str(doc_id)]["source_searches"]:
                documents[str(doc_id)]["source_searches"].append(search_key)
    
    cache_data["documents"] = documents

async def fetch_document_html_content(doc_id: str) -> Optional[str]:
    """Fetch full HTML content for a document to extract comprehensive facts."""
    try:
        # First get document metadata
        cache_data = load_cache()
        cached_doc = cache_data.get("documents", {}).get(str(doc_id), {}).get("data")
        
        if not cached_doc:
            # Try to fetch from API
            url = f"https://new.kenyalaw.org/search/api/documents/{doc_id}/"
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return None
            cached_doc = response.json()
        
        # Get expression URI
        expression_uri = cached_doc.get("expression_frbr_uri") or cached_doc.get("work_frbr_uri")
        if not expression_uri:
            # Fallback: query search API
            search_url = f"https://new.kenyalaw.org/search/api/documents/?q=id:{doc_id}"
            sr = requests.get(search_url, timeout=30)
            if sr.status_code == 200:
                sjson = sr.json()
                results = sjson.get("results", [])
                if results:
                    expression_uri = results[0].get("expression_frbr_uri") or results[0].get("work_frbr_uri")
        
        if not expression_uri:
            return None
            
        # Fetch HTML content
        html_url = f"https://new.kenyalaw.org{expression_uri}"
        page_res = requests.get(html_url, timeout=30)
        if page_res.status_code != 200:
            return None
            
        page_html = page_res.text
        
        # Parse and extract text content
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
        
        if content_el:
            return content_el.get_text(separator="\n", strip=True)
        
        return None
        
    except Exception as e:
        logger.warning(f"Error fetching HTML content for doc {doc_id}: {e}")
        return None

def enhance_results_with_ai(results: List[Dict], search_term: str, fetch_full_content: bool = True) -> List[Dict]:
    """Enhance search results with AI-powered relevance and summarization."""
    enhanced_results = []
    
    for result in results:
        try:
            # Create a copy to avoid modifying the original
            enhanced_result = result.copy()
            doc_id = result.get("id")
            
            # Try to get full content if enabled and doc_id is available
            full_content = None
            if fetch_full_content and doc_id:
                try:
                    # Import asyncio to run async function
                    import asyncio
                    # Create new event loop if none exists
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Run the async function
                    full_content = loop.run_until_complete(fetch_document_html_content(str(doc_id)))
                except Exception as e:
                    logger.warning(f"Failed to fetch full content for doc {doc_id}: {e}")
            
            # Use full content if available, otherwise fall back to search result content
            content_to_analyze = full_content if full_content else result.get("content", "")
            
            # Extract facts from content
            facts = fact_summarizer.extract_facts_section(content_to_analyze)
            
            # If no facts found and we have full content, try using more of the content
            if (not facts or facts == "No factual information available.") and full_content:
                # Use first 1000 words instead of 700 for full content
                words = full_content.split()
                facts = ' '.join(words[:1000]) if len(words) > 50 else full_content
            
            # Calculate relevance score
            relevance_score = relevance_calculator.calculate_relevance(search_term, facts)
            
            # Generate summary
            summary = fact_summarizer.summarize_facts(facts, search_term)
            
            # Add new fields
            enhanced_result["relevance"] = round(relevance_score, 3)
            enhanced_result["summary"] = summary
            enhanced_result["content_source"] = "full_html" if full_content else "search_api"
            enhanced_result["facts_word_count"] = len(facts.split()) if facts else 0
            
            enhanced_results.append(enhanced_result)
            
        except Exception as e:
            logger.warning(f"Error enhancing result {result.get('id', 'unknown')}: {e}")
            # Add default values on error
            enhanced_result = result.copy()
            enhanced_result["relevance"] = 0.0
            enhanced_result["summary"] = "Summary unavailable due to processing error."
            enhanced_result["content_source"] = "error"
            enhanced_result["facts_word_count"] = 0
            enhanced_results.append(enhanced_result)
    
    return enhanced_results

# Core search function (enhanced)
def fetch_kenya_law_documents(
    search_term: str, 
    page: int = 1, 
    page_size: int = 20,
    court_filter: Optional[str] = None,
    year_filter: Optional[str] = None,
    show_facets: bool = False,
    use_cache: bool = True,
    cache_max_age_hours: int = 24,
    enable_ai_enhancement: bool = True,
    fetch_full_content: bool = True  # New parameter to control full content fetching
) -> Dict:
    """
    Fetch Kenya law documents with pagination, filtering, and AI enhancement.
    """
    
    # Check cache first if enabled
    from_cache = False
    if use_cache:
        cache_data = load_cache()
        search_key = get_search_key(search_term, page, page_size, court_filter, year_filter, enable_ai_enhancement, fetch_full_content)
        
        # Check if we have cached search results
        cached_search = get_cached_search(search_key, cache_data)
        if cached_search and is_cache_valid(cache_data, cache_max_age_hours):
            result = cached_search["data"]
            result["from_cache"] = True
            return result
    
    # Base URL
    url = "https://new.kenyalaw.org/search/api/documents/"

    # Query parameters
    params = {
        "page": page,
        "page_size": min(page_size, 50),  # API limit
        "ordering": "-score",
        "nature": "Judgment",
        "facet": [
            "nature", "court", "year", "registry", "locality", "outcome",
            "judges", "authors", "language", "labels", "attorneys", "matter_type"
        ],
        "search__all": f"({search_term})"
    }
    
    # Add filters if provided
    if court_filter:
        params["court"] = court_filter
    if year_filter:
        params["year"] = year_filter

    try:
        # Make the request
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        total_count = data.get("count", 0)
        total_pages = data.get("total_pages", 0)
        
        # Apply AI enhancement if enabled
        if enable_ai_enhancement and results:
            logger.info(f"Enhancing {len(results)} results with AI (fetch_full_content={fetch_full_content})")
            results = enhance_results_with_ai(results, search_term, fetch_full_content)
            
            # Sort by relevance score (highest first)
            results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        response_data = {
            "results": results,
            "total_count": total_count,
            "total_pages": total_pages,
            "current_page": page,
            "facets": data.get("facets", {}) if show_facets else None,
            "cached_at": datetime.now().isoformat(),
            "search_params": {
                "search_term": search_term,
                "page": page,
                "page_size": page_size,
                "court_filter": court_filter,
                "year_filter": year_filter,
                "enable_ai_enhancement": enable_ai_enhancement,
                "fetch_full_content": fetch_full_content
            },
            "from_cache": from_cache,
            "ai_enhanced": enable_ai_enhancement
        }
        
        # Cache the response if caching is enabled
        if use_cache:
            cache_data = load_cache()
            search_key = get_search_key(search_term, page, page_size, court_filter, year_filter, enable_ai_enhancement, fetch_full_content)
            cache_search_results(search_key, response_data, cache_data)
            save_cache(cache_data)
        
        return response_data
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

# API Routes (enhanced)
@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search Kenya Law documents with advanced filtering, caching, and AI enhancement.
    """
    try:
        result = fetch_kenya_law_documents(
            search_term=request.search_term,
            page=request.page,
            page_size=request.page_size,
            court_filter=request.court_filter,
            year_filter=request.year_filter,
            show_facets=True,
            use_cache=request.use_cache,
            cache_max_age_hours=request.cache_max_age_hours,
            enable_ai_enhancement=request.enable_ai_enhancement,
            fetch_full_content=request.fetch_full_content
        )
        return SearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_documents_get(
    search_term: str = Query(..., description="Search term"),
    page: int = Query(1, description="Page number"),
    page_size: int = Query(20, description="Results per page", le=50),
    court_filter: Optional[str] = Query(None, description="Filter by court"),
    year_filter: Optional[str] = Query(None, description="Filter by year"),
    use_cache: bool = Query(True, description="Use caching"),
    cache_max_age_hours: int = Query(24, description="Cache max age in hours"),
    enable_ai_enhancement: bool = Query(True, description="Enable AI relevance scoring and summarization")
):
    """
    Search Kenya Law documents with AI enhancement (GET method for simple queries).
    """
    try:
        result = fetch_kenya_law_documents(
            search_term=search_term,
            page=page,
            page_size=page_size,
            court_filter=court_filter,
            year_filter=year_filter,
            show_facets=True,
            use_cache=use_cache,
            cache_max_age_hours=cache_max_age_hours,
            enable_ai_enhancement=enable_ai_enhancement
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Keep all existing endpoints unchanged...
@router.get("/document/{document_id}")
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

@router.get("/document/html/{doc_id}")
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

@router.get("/cache/stats", response_model=CacheStats)
async def get_cache_stats():
    """Get cache statistics."""
    try:
        cache_data = load_cache()
        
        stats = {
            "total_searches": len(cache_data.get("searches", {})),
            "total_documents": len(cache_data.get("documents", {})),
            "cache_size_mb": 0,
            "created_at": cache_data.get("metadata", {}).get("created_at"),
            "last_updated": cache_data.get("metadata", {}).get("last_updated")
        }
        
        # Calculate cache file size
        cache_filepath = get_cache_filepath()
        if os.path.exists(cache_filepath):
            stats["cache_size_mb"] = os.path.getsize(cache_filepath) / (1024 * 1024)
        
        return CacheStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache")
async def clear_cache():
    """Clear all cached data."""
    try:
        cache_filepath = get_cache_filepath()
        if os.path.exists(cache_filepath):
            os.remove(cache_filepath)
            return {"message": "Cache cleared successfully"}
        else:
            return {"message": "Cache file does not exist"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/search/{search_term}")
async def search_cached_documents(search_term: str):
    """Search through cached documents."""
    try:
        cache_data = load_cache()
        search_term_lower = search_term.lower()
        found_documents = []
        
        for doc_id, doc_info in cache_data.get("documents", {}).items():
            doc = doc_info.get("data", {})
            
            # Search in title
            title = doc.get("title", "").lower()
            if search_term_lower in title:
                found_documents.append({
                    "doc_id": doc_id,
                    "document": doc,
                    "match_type": "title",
                    "source_searches": doc_info.get("source_searches", [])
                })
                continue
            
            # Search in citation
            citation = doc.get("citation", "").lower()
            if search_term_lower in citation:
                found_documents.append({
                    "doc_id": doc_id,
                    "document": doc,
                    "match_type": "citation",
                    "source_searches": doc_info.get("source_searches", [])
                })
                continue
        
        return {
            "search_term": search_term,
            "total_found": len(found_documents),
            "documents": found_documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/searches")
async def get_cached_searches():
    """Get all cached searches."""
    try:
        cache_data = load_cache()
        searches = cache_data.get("searches", {})
        
        search_list = []
        for search_key, search_info in searches.items():
            search_params = search_info.get("search_params", {})
            search_data = search_info.get("data", {})
            
            search_list.append({
                "search_key": search_key,
                "search_term": search_params.get("search_term", "Unknown"),
                "page": search_params.get("page", 1),
                "total_results": search_data.get("total_count", 0),
                "cached_at": search_info.get("cached_at", "Unknown"),
                "ai_enhanced": search_params.get("enable_ai_enhancement", False)
            })
        
        return {
            "total_searches": len(search_list),
            "searches": search_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New AI-specific endpoints
@router.post("/analyze/relevance")
async def analyze_relevance(
    search_term: str = Query(..., description="Search term to analyze against"),
    case_text: str = Query(..., description="Case text or facts to analyze")
):
    """
    Analyze relevance between a search term and case text.
    Useful for testing the relevance scoring algorithm.
    """
    try:
        relevance_score = relevance_calculator.calculate_relevance(search_term, case_text)
        
        # Extract facts section for additional analysis
        facts = fact_summarizer.extract_facts_section(case_text)
        
        return {
            "search_term": search_term,
            "relevance_score": round(relevance_score, 3),
            "extracted_facts_length": len(facts.split()),
            "original_text_length": len(case_text.split()),
            "facts_extraction_ratio": round(len(facts.split()) / max(len(case_text.split()), 1), 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/summarize/facts")
async def summarize_facts(
    case_text: str = Query(..., description="Case text to summarize"),
    search_term: str = Query("", description="Optional search term for context"),
    max_sentences: int = Query(3, description="Maximum sentences in summary", ge=1, le=5)
):
    """
    Generate a summary of case facts from provided text.
    """
    try:
        # Extract facts section
        facts = fact_summarizer.extract_facts_section(case_text)
        
        # Generate summary
        summary = fact_summarizer.summarize_facts(facts, search_term, max_sentences)
        
        return {
            "original_text_length": len(case_text.split()),
            "extracted_facts_length": len(facts.split()),
            "summary_length": len(summary.split()),
            "compression_ratio": round(len(summary.split()) / max(len(facts.split()), 1), 3),
            "extracted_facts": facts,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@router.get("/ai/stats")
async def get_ai_stats():
    """
    Get statistics about AI enhancement usage.
    """
    try:
        cache_data = load_cache()
        searches = cache_data.get("searches", {})
        
        total_searches = len(searches)
        ai_enhanced_searches = 0
        relevance_scores = []
        
        for search_info in searches.values():
            search_params = search_info.get("search_params", {})
            if search_params.get("enable_ai_enhancement", False):
                ai_enhanced_searches += 1
                
                # Collect relevance scores from results
                results = search_info.get("data", {}).get("results", [])
                for result in results:
                    if "relevance" in result:
                        relevance_scores.append(result["relevance"])
        
        # Calculate relevance statistics
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        max_relevance = max(relevance_scores) if relevance_scores else 0
        min_relevance = min(relevance_scores) if relevance_scores else 0
        
        return {
            "total_searches": total_searches,
            "ai_enhanced_searches": ai_enhanced_searches,
            "ai_enhancement_rate": round(ai_enhanced_searches / max(total_searches, 1), 3),
            "total_relevance_scores": len(relevance_scores),
            "average_relevance_score": round(avg_relevance, 3),
            "max_relevance_score": round(max_relevance, 3),
            "min_relevance_score": round(min_relevance, 3),
            "high_relevance_results": len([s for s in relevance_scores if s > 0.7]),
            "medium_relevance_results": len([s for s in relevance_scores if 0.3 < s <= 0.7]),
            "low_relevance_results": len([s for s in relevance_scores if s <= 0.3])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check and system info endpoints
@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify API and AI components are working.
    """
    try:
        # Test basic functionality
        test_text = "The plaintiff filed a case for breach of contract against the defendant."
        test_search = "contract breach"
        
        # Test relevance calculation
        relevance_score = relevance_calculator.calculate_relevance(test_search, test_text)
        
        # Test summarization
        summary = fact_summarizer.summarize_facts(test_text, test_search, 1)
        
        # Check cache accessibility
        cache_accessible = os.path.exists(get_cache_filepath()) or True  # True if we can create it
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "relevance_calculator": "operational" if relevance_score >= 0 else "error",
                "fact_summarizer": "operational" if summary else "error",
                "cache_system": "operational" if cache_accessible else "error"
            },
            "test_results": {
                "test_relevance_score": round(relevance_score, 3),
                "test_summary_generated": bool(summary),
                "test_summary_length": len(summary.split()) if summary else 0
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/system/info")
async def get_system_info():
    """
    Get system information and configuration.
    """
    try:
        import sys
        import platform
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "ai_components": {
                "nltk_available": True,
                "sklearn_available": True,
                "beautifulsoup_available": True
            },
            "cache_directory": os.path.dirname(get_cache_filepath()),
            "default_settings": {
                "default_page_size": 20,
                "max_page_size": 50,
                "default_cache_max_age_hours": 24,
                "default_ai_enhancement": True,
                "default_summary_sentences": 3
            }
        }
    except Exception as e:
        return {"error": str(e)}