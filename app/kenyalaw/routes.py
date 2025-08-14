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

router = APIRouter(prefix="/kenyalaw", tags=["Kenya Law"])

# Pydantic models for request/response
class SearchRequest(BaseModel):
    search_term: str
    page: int = 1
    page_size: int = 20
    court_filter: Optional[str] = None
    year_filter: Optional[str] = None
    use_cache: bool = True
    cache_max_age_hours: int = 24

class SearchResponse(BaseModel):
    results: List[Dict]
    total_count: int
    total_pages: int
    current_page: int
    facets: Optional[Dict] = None
    cached_at: str
    search_params: Dict
    from_cache: bool = False

class CacheStats(BaseModel):
    total_searches: int
    total_documents: int
    cache_size_mb: float
    created_at: Optional[str]
    last_updated: Optional[str]

# Cache management functions
def get_cache_filepath() -> str:
    """Get the filepath for the main cache file."""
    cache_dir = "kenyalaw_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return os.path.join(cache_dir, "kenyalaw_cache.json")

def get_search_key(search_term: str, page: int, page_size: int, court_filter: Optional[str] = None, year_filter: Optional[str] = None) -> str:
    """Generate a unique search key for the search parameters."""
    cache_string = f"{search_term}_{page}_{page_size}_{court_filter}_{year_filter}"
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
        print(f"Failed to save cache: {e}")

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

# Core search function
def fetch_kenya_law_documents(
    search_term: str, 
    page: int = 1, 
    page_size: int = 20,
    court_filter: Optional[str] = None,
    year_filter: Optional[str] = None,
    show_facets: bool = False,
    use_cache: bool = True,
    cache_max_age_hours: int = 24
) -> Dict:
    """
    Fetch Kenya law documents with pagination and filtering.
    """
    
    # Check cache first if enabled
    from_cache = False
    if use_cache:
        cache_data = load_cache()
        search_key = get_search_key(search_term, page, page_size, court_filter, year_filter)
        
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
                "year_filter": year_filter
            },
            "from_cache": from_cache
        }
        
        # Cache the response if caching is enabled
        if use_cache:
            cache_data = load_cache()
            search_key = get_search_key(search_term, page, page_size, court_filter, year_filter)
            cache_search_results(search_key, response_data, cache_data)
            save_cache(cache_data)
        
        return response_data
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

# API Routes
@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search Kenya Law documents with advanced filtering and caching.
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
            cache_max_age_hours=request.cache_max_age_hours
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
    cache_max_age_hours: int = Query(24, description="Cache max age in hours")
):
    """
    Search Kenya Law documents (GET method for simple queries).
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
            cache_max_age_hours=cache_max_age_hours
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/document/{document_id}")
async def get_document(document_id: str):
    """
    Get a specific document by ID from cache or API.
    """
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
        
        # If not in cache, try to fetch from API (this might not work as we discovered)
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
        
@router.get("/document/{document_id}")
async def get_document(document_id: str):
    """
    Get a specific document by ID from cache or API.
    """
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
        
        # If not in cache, try to fetch from API (this might not work as we discovered)
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
    """
    Use existing get_document() to get metadata, resolve the AKN expression URI,
    fetch the full AKN HTML page, and return:
      - full inner HTML of the judgment body (content_html)
      - cleaned plain text (content_text)
      - metadata (title, date, court, registry, case_number, judges, attorneys)
      - citations (links/text pointing to other judgments)
      - attachments (S3 URLs)
    """
    try:
        # 1) Re-use the upper method to get metadata (cache-aware)
        meta_resp = await get_document(doc_id)
        # meta_resp is expected to be a dict with key "document"
        metadata = None
        if isinstance(meta_resp, dict):
            metadata = meta_resp.get("document") or meta_resp  # robust handling
        else:
            metadata = meta_resp

        if not metadata:
            raise HTTPException(status_code=404, detail="Document metadata not found")

        # 2) Resolve expression_frbr_uri (try metadata, then fallback to search)
        expression_uri = metadata.get("expression_frbr_uri") or metadata.get("work_frbr_uri")
        if not expression_uri:
            # fallback: query search API for id
            search_url = f"https://new.kenyalaw.org/search/api/documents/?q=id:{doc_id}"
            sr = requests.get(search_url, timeout=30)
            sr.raise_for_status()
            sjson = sr.json()
            results = sjson.get("results", [])
            if not results:
                raise HTTPException(status_code=404, detail="Document expression URI not found on search fallback")
            expression_uri = results[0].get("expression_frbr_uri") or results[0].get("work_frbr_uri")

        if not expression_uri:
            raise HTTPException(status_code=404, detail="No expression URI available for document")

        # 3) Fetch the AKN HTML page
        html_url = f"https://new.kenyalaw.org{expression_uri}"
        page_res = requests.get(html_url, timeout=30)
        page_res.raise_for_status()
        page_html = page_res.text

        # 4) Parse HTML and isolate the judgment body
        soup = BeautifulSoup(page_html, "html.parser")

        # Remove global chrome we don't want (but keep the judgment body)
        for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
            tag.decompose()

        # Prefer the main judgment element(s)
        content_el = (
            soup.select_one("#document_content") or
            soup.select_one("div.content__html") or
            soup.select_one("div.document-content__inner") or
            soup.select_one("div.document-content") or
            soup.select_one("div.content-and-enrichments__inner") or
            soup.body  # worst-case fallback: whole body
        )

        # full inner HTML of the judgment body (preserves markup)
        content_html = content_el.decode_contents() if content_el else ""
        # cleaned plain text with reasonable separators
        content_text = content_el.get_text(separator="\n", strip=True) if content_el else soup.get_text("\n", strip=True)

        # 5) Extract metadata from the HTML (dl.document-metadata-list)
        html_metadata = {}
        for dl in soup.select("dl.document-metadata-list"):
            dts = dl.find_all("dt")
            for dt in dts:
                dd = dt.find_next_sibling("dd")
                if dd:
                    key = dt.get_text(strip=True)
                    val = " ".join(dd.stripped_strings)
                    # normalize keys to snake_case
                    html_metadata[key.lower().replace(" ", "_")] = val

        # Merge API metadata + HTML metadata (HTML metadata overrides)
        merged_meta = { **(metadata or {}), **html_metadata }

        # 6) Attachments: find S3 / attachments links
        attachments = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if "kenyalaw-website-media" in href or "/attachments/" in href:
                attachments.append({
                    "text": a.get_text(strip=True),
                    "url": href
                })
        # dedupe
        seen = set()
        attachments = [a for a in attachments if (a["url"] not in seen and not seen.add(a["url"]))]

        # 7) Citations: anchors pointing to other judgments (akn paths or /judgments/)
        citations = []
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if "/akn/ke/judgment/" in href or "/judgments/" in href or "KEHC" in a.get_text():
                txt = a.get_text(strip=True)
                if txt:
                    citations.append(txt)
        # Also extract bracket-style citations from the text as fallback
        bracket_cites = re.findall(r"\[\d{4}\]\s*[A-Z]{2,}[^\n\(\)]+(?:\([A-Z]{3,}\))?", content_text)
        citations = list(dict.fromkeys(citations + bracket_cites))  # preserve order, dedupe

        # 8) Optional: try to split into common sections (BACKGROUND / ANALYSIS / RULING)
        sections = {}
        headings = re.split(r"\n([A-Z][A-Z\s]{3,})\n", content_text)
        if len(headings) > 1:
            # headings pattern yields [pre, H1, section1, H2, section2, ...]
            pre = headings[0].strip()
            if pre:
                sections["preface"] = pre
            for i in range(1, len(headings), 2):
                h = headings[i].strip().lower()
                body = headings[i+1].strip() if i+1 < len(headings) else ""
                sections[h] = body

        # 9) Final response
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
    """
    Get cache statistics.
    """
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
    """
    Clear all cached data.
    """
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
    """
    Search through cached documents.
    """
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
    """
    Get all cached searches.
    """
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
                "cached_at": search_info.get("cached_at", "Unknown")
            })
        
        return {
            "total_searches": len(search_list),
            "searches": search_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


