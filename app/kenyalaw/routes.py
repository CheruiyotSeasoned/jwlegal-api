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

# Traditional AI Enhancement Classes (Your existing classes with improvements)
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

# New GPT Enhancement Classes
class GPTRelevanceCalculator:
    """Use GPT for semantic relevance when traditional methods aren't sufficient."""
    
    def __init__(self, openai_api_key: str):
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=openai_api_key)
            self.available = True
        except ImportError:
            logger.warning("OpenAI package not available for GPT features")
            self.available = False
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            self.available = False
            
    async def calculate_gpt_relevance(self, search_term: str, case_facts: str, case_title: str = "") -> Dict:
        """Use GPT to analyze relevance and provide reasoning."""
        
        if not self.available:
            return {"relevance_score": 0.0, "reasoning": "GPT unavailable", "source": "error"}
        
        prompt = f"""Analyze the relevance of this Kenyan legal case to the search query.
        Search Query: "{search_term}"
        Case Title: "{case_title}"
        Case Facts: "{case_facts[:2000]}"
       
        Rate relevance from 0.0 to 1.0:
        - 0.8-1.0: Highly relevant (direct legal precedent, same principles)
        - 0.5-0.79: Moderately relevant (related legal area, similar facts)
        - 0.2-0.49: Somewhat relevant (tangential legal connection)
        - 0.0-0.19: Not relevant
        Provide your score and brief reasoning.
        Respond ONLY in JSON format:
        {{"relevance_score": 0.85, "reasoning": "Brief explanation"}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Kenyan legal research assistant. Provide accurate relevance assessments in JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                "relevance_score": float(result.get("relevance_score", 0.0)),
                "reasoning": result.get("reasoning", "No reasoning provided"),
                "source": "gpt"
            }
            
        except Exception as e:
            logger.warning(f"GPT relevance calculation failed: {e}")
            return {"relevance_score": 0.0, "reasoning": "GPT analysis failed", "source": "error"}

class GPTSummarizer:
    """Use GPT for advanced case summarization."""
    
    def __init__(self, openai_api_key: str):
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=openai_api_key)
            self.available = True
        except ImportError:
            logger.warning("OpenAI package not available for GPT features")
            self.available = False
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            self.available = False
    
    async def summarize_case_with_gpt(self, case_content: str, search_term: str = "") -> str:
        """Generate a comprehensive case summary using GPT."""
        
        if not self.available:
            return "GPT summarization unavailable."
        
        # Chunk content if too long
        if len(case_content) > 8000:  # Rough token limit
            chunks = [case_content[i:i+8000] for i in range(0, len(case_content), 8000)]
            summaries = []
            
            for chunk in chunks[:3]:  # Limit to 3 chunks for cost control
                chunk_summary = await self._summarize_chunk(chunk, search_term)
                if chunk_summary:
                    summaries.append(chunk_summary)
            
            if summaries:
                return await self._synthesize_summaries(summaries, search_term)
            else:
                return "Case summarization failed."
        else:
            return await self._summarize_chunk(case_content, search_term)
    
    async def _summarize_chunk(self, content: str, search_term: str) -> str:
        """Summarize a single chunk of content."""
        
        prompt = f"""Summarize this Kenyan legal case content concisely.

        Search Context: "{search_term}"
        Case Content: "{content}"

        Provide a 2-3 sentence summary covering:
        1. Key facts and parties involved
        2. Main legal issue(s)
        3. Court's decision/ruling

        Focus on elements most relevant to the search context."""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a legal case summarization assistant. Be concise and accurate."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"GPT chunk summarization failed: {e}")
            return ""
    
    async def _synthesize_summaries(self, summaries: List[str], search_term: str) -> str:
        """Combine multiple chunk summaries into a coherent whole."""
        
        combined = "\n\n".join([f"Part {i+1}: {summary}" for i, summary in enumerate(summaries)])
        
        prompt = f"""Combine these partial case summaries into a single coherent summary.

        Search Context: "{search_term}"
        Partial Summaries:
        {combined}

        Create a unified 3-4 sentence summary that captures the complete case."""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Synthesize legal case summaries clearly and concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"GPT summary synthesis failed: {e}")
            return summaries[0] if summaries else "Synthesis failed."

# Rate limiter for external API calls
class RateLimiter:
    """Simple rate limiter to prevent overwhelming external APIs."""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        # Wait if we're at the limit
        if len(self.requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.requests.append(now)

# Global rate limiter for Kenya Law API
kenya_law_rate_limiter = RateLimiter(max_requests=10, time_window=60)

# Hybrid Enhancement Engine
class HybridEnhancer:
    """Combine traditional NLP with selective GPT enhancement."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.traditional_relevance = RelevanceCalculator()
        self.traditional_summarizer = FactSummarizer()
        
        # Initialize GPT components if API key is provided
        if openai_api_key:
            self.gpt_relevance = GPTRelevanceCalculator(openai_api_key)
            self.gpt_summarizer = GPTSummarizer(openai_api_key)
        else:
            self.gpt_relevance = None
            self.gpt_summarizer = None
        
        self.stats = {
            "traditional_enhanced": 0,
            "gpt_enhanced": 0,
            "hybrid_enhanced": 0,
            "full_content_fetched": 0,
            "gpt_failures": 0
        }
    
    async def enhance_results_hybrid(
        self, 
        results: List[Dict], 
        search_term: str, 
        enhancement_level: str = "traditional",
        gpt_rerank_top_n: int = 10,
        fetch_full_content: bool = False
        ) -> tuple[List[Dict], Dict]:
        """Enhanced results using hybrid approach."""
        
        if enhancement_level == "none":
            return results, self.stats
        
        enhanced_results = []
        # for result in results:
        #     print(result["title"])
        #     print(result["full_content"][:500])  # preview first 500 chars
        #     print(result["content_source"])

        
        # Step 1: Traditional enhancement for all results
        for result in results:
            enhanced_result = await self._enhance_with_traditional(result, search_term, fetch_full_content)
            enhanced_results.append(enhanced_result)
            self.stats["traditional_enhanced"] += 1
        
        # Step 2: Sort by traditional relevance
        enhanced_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Step 3: Apply GPT enhancement selectively
        if enhancement_level in ["gpt", "hybrid"] and self.gpt_relevance:
            top_results = enhanced_results[:gpt_rerank_top_n]
            
            for i, result in enumerate(top_results):
                try:
                    # GPT relevance analysis
                    highlights = ""
                    if "highlight" in result and "content" in result["highlight"]:
                        highlights = "\n".join(result["highlight"]["content"])
                    else:
                        highlights = result.get("full_content")
                    # GPT relevance analysis (summary + highlights)
                    gpt_analysis = await self.gpt_relevance.calculate_gpt_relevance(
                        search_term=search_term,
                        case_facts=highlights,
                        case_title=result.get("title", "")
                    )
                    
                    if enhancement_level == "gpt":
                        # Replace traditional score with GPT score
                        result["relevance"] = gpt_analysis["relevance_score"]
                        result["relevance_source"] = "gpt"
                        self.stats["gpt_enhanced"] += 1
                    elif enhancement_level == "hybrid":
                        # Combine scores (weighted average)
                        traditional_score = result.get("relevance", 0.0)
                        gpt_score = gpt_analysis["relevance_score"]
                        combined_score = (traditional_score * 0.4) + (gpt_score * 0.6)
                        result["relevance"] = round(combined_score, 3)
                        result["relevance_source"] = "hybrid"
                        self.stats["hybrid_enhanced"] += 1
                    
                    result["gpt_reasoning"] = gpt_analysis["reasoning"]
                    
                    # Optional: Enhanced summarization with GPT
                    if self.gpt_summarizer and result.get("content_source") == "full_html":
                        enhanced_summary = await self.gpt_summarizer.summarize_case_with_gpt(
                            case_content=result.get("full_content", ""),
                            search_term=search_term
                        )
                        if enhanced_summary and enhanced_summary != "GPT summarization unavailable.":
                            result["gpt_summary"] = enhanced_summary
                    
                except Exception as e:
                    logger.warning(f"GPT enhancement failed for result {i}: {e}")
                    result["gpt_reasoning"] = "GPT analysis unavailable"
                    self.stats["gpt_failures"] += 1
            
            # Re-sort after GPT enhancement
            enhanced_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        return enhanced_results, self.stats


    async def summarize_only(
        results: List[Dict],
        search_term: str
    ) -> List[Dict]:
        """
        Summarizes search/document results without reranking or enhancement.
        Returns a list of dicts with a 'gpt_summary' field.
        """
        if not results:
            return []

        content_blocks = []
        for i, doc in enumerate(results[:10], start=1):
            title = doc.get("title") or doc.get("name") or f"Result {i}"
            snippet = doc.get("snippet") or doc.get("content") or ""
            content_blocks.append(f"{i}. {title}\n{snippet}")

        combined_text = "\n\n".join(content_blocks)

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who summarizes documents clearly and concisely."},
                {"role": "user", "content": f"Summarize the following results about '{search_term}':\n\n{combined_text}"}
            ],
            temperature=0.4,
            max_tokens=600
        )

        summary_text = response.choices[0].message.content.strip()

        summarized = []
        for doc in results:
            summarized.append({**doc, "gpt_summary": summary_text})

        return summarized

    async def _enhance_with_traditional(self, result: Dict, search_term: str, fetch_full_content: bool) -> Dict:
        """Apply traditional NLP enhancement."""
        enhanced_result = result.copy()
        doc_id = result.get("id")
        
        # Get content
        full_content = None
        if fetch_full_content and doc_id:
            try:
                await kenya_law_rate_limiter.acquire()  # Respect rate limits
                full_content = await fetch_document_html_content(str(doc_id))
                if full_content:
                    self.stats["full_content_fetched"] += 1
                    enhanced_result["full_content"] = full_content  # Store for potential GPT use
            except Exception as e:
                logger.warning(f"Failed to fetch full content for doc {doc_id}: {e}")
        
        content_to_analyze = full_content if full_content else result.get("content", "")
        
        # Extract facts and calculate relevance
        facts = self.traditional_summarizer.extract_facts_section(content_to_analyze)
        if (not facts or facts == "No factual information available.") and full_content:
            words = full_content.split()
            facts = ' '.join(words[:1000]) if len(words) > 50 else full_content
        
        relevance_score = self.traditional_relevance.calculate_relevance(search_term, facts)
        summary = self.traditional_summarizer.summarize_facts(facts, search_term)
        
        # Add enhancement data
        enhanced_result.update({
            "relevance": round(relevance_score, 3),
            "summary": summary,
            "content_source": "full_html" if full_content else "search_api",
            "facts_word_count": len(facts.split()) if facts else 0,
            "relevance_source": "traditional"
        })
        
        return enhanced_result

# Initialize AI components
traditional_relevance_calculator = RelevanceCalculator()
traditional_fact_summarizer = FactSummarizer()

# Cache management functions (unchanged from your original)
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

async def fetch_kenya_law_documents_hybrid(
    search_term: str,
    page: int = 1,
    page_size: int = 20,
    court_filter: Optional[str] = None,
    year_filter: Optional[str] = None,
    use_cache: bool = True,
    cache_max_age_hours: int = 24,
    ai_enhancement_level: str = "gpt",  # "none" | "traditional" | "deep"
    gpt_rerank_top_n: int = 10,
    fetch_full_content: bool = True,
    openai_api_key: Optional[str] = None
    ) -> Dict:
        """Enhanced fetch function with hybrid AI capabilities + logging."""
        
        logger.info(
            f"Fetching Kenya Law documents | term='{search_term}', page={page}, "
            f"page_size={page_size}, court={court_filter}, year={year_filter}, "
            f"ai_level={ai_enhancement_level}, cache={use_cache}"
        )

    # Cache check
        from_cache = False
        if use_cache:
            cache_data = load_cache()
            search_key = get_search_key(
                search_term, page, page_size, court_filter, year_filter,
                ai_enhancement_level != "none", fetch_full_content
            )
            cached_search = get_cached_search(search_key, cache_data)

            if cached_search and is_cache_valid(cache_data, cache_max_age_hours):
                logger.info(f"Cache hit for key={search_key}")
                result = cached_search["data"]
                result["from_cache"] = True
                return result
            else:
                logger.info(f"No valid cache found for key={search_key}")

            # API call
            url = "https://new.kenyalaw.org/search/api/documents/"
            params = {
                "page": page,
                "page_size": min(page_size, 50),
                "ordering": "-score",
                "nature": "Judgment",
                "facet": [
                    "nature", "court", "year", "registry", "locality", "outcome",
                    "judges", "authors", "language", "labels", "attorneys", "matter_type"
                ],
                "search__all": f"({search_term})"
            }

            if court_filter:
                params["court"] = court_filter
            if year_filter:
                params["year"] = year_filter

            try:
                logger.info(f"Calling Kenya Law API: {url} with params={params}")
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()
                results = data.get("results", [])
                logger.info(f"API returned {len(results)} results (total={data.get('count', 0)})")

                # Apply hybrid enhancement
                if ai_enhancement_level != "none" and results:
                    logger.info(f"Enhancing results with AI | level={ai_enhancement_level}, top_n={gpt_rerank_top_n}")
                    enhancer = HybridEnhancer(openai_api_key)
                    results = await enhancer.enhance_results_hybrid(
                        results=results,
                        search_term=search_term,
                        enhancement_level="traditional",
                        gpt_rerank_top_n=gpt_rerank_top_n,
                        fetch_full_content=False
                    )
                    logger.info(f"Enhancement complete | traditional={enhancer.stats['traditional_enhanced']}, "
                                f"gpt={enhancer.stats['gpt_enhanced']}, hybrid={enhancer.stats['hybrid_enhanced']}, "
                                f"full_content_fetched={enhancer.stats['full_content_fetched']}, gpt_failures={enhancer.stats['gpt_failures']}")
                    if results and isinstance(results[0], dict):
                        logger.info(f"Final top result relevance: {results[0].get('relevance', 'N/A')}")
                    elif results and isinstance(results[0], list) and len(results[0]) > 0:
                        logger.info(f"Final top result relevance>: {results[0][0].get('relevance', 'N/A')}")
                    else:
                        logger.info("No valid results found for relevance logging")
                    
                response_data = {
                    "results": results,
                    "total_count": data.get("count", 0),
                    "total_pages": data.get("total_pages", 0),
                    "current_page": page,
                    "facets": data.get("facets", {}),
                    "cached_at": datetime.now().isoformat(),
                    "search_params": {
                        "search_term": search_term,
                        "page": page,
                        "page_size": page_size,
                        "ai_enhancement_level": ai_enhancement_level,
                        "gpt_rerank_top_n": gpt_rerank_top_n,
                    },
                    "from_cache": from_cache,
                    "ai_enhanced": ai_enhancement_level != "none",
                }

                # Cache the response
                if use_cache:
                    cache_data = load_cache()
                    search_key = get_search_key(
                        search_term,
                        page,
                        page_size,
                        court_filter,
                        year_filter,
                        ai_enhanced=True,
                        fetch_full=False,
                        enhancement_level="hybrid"  # or pass dynamically if you support levels
                    )
                    cache_search_results(search_key, response_data, cache_data)
                    save_cache(cache_data)

                return response_data

            except Exception as e:
                logger.error(f"Request failed for term='{search_term}': {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
# API Routes (enhanced)
@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search Kenya Law documents with advanced filtering, caching, and AI enhancement.
    """
    try:
        result = await fetch_kenya_law_documents_hybrid(
            search_term=request.search_term,
            page=request.page,
            page_size=request.page_size,
            court_filter=request.court_filter,
            year_filter=request.year_filter,
            use_cache=request.use_cache,
            cache_max_age_hours=request.cache_max_age_hours,
            ai_enhancement_level=request.enable_ai_enhancement,  # match parameter name
            gpt_rerank_top_n=request.gpt_rerank_top_n if hasattr(request, "gpt_rerank_top_n") else 10,
            fetch_full_content=request.fetch_full_content,
            openai_api_key=api_key  # load from env in production
        )
        return SearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_documents_get(
    search_term: str = Query(..., description="Search term"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, le=50, description="Results per page"),
    court_filter: Optional[str] = Query(None, description="Filter by court"),
    year_filter: Optional[str] = Query(None, description="Filter by year"),
    use_cache: bool = Query(True, description="Use caching"),
    cache_max_age_hours: int = Query(24, ge=1, description="Cache max age in hours"),
    enable_ai_enhancement: bool = Query(True, description="Enable AI relevance scoring and summarization")
):
    """
    Search Kenya Law documents with optional AI-powered relevance scoring and summarization.
    """
    try:
        result = await fetch_kenya_law_documents_hybrid(
        search_term=search_term,
        page=page,
        page_size=page_size,
        court_filter=court_filter,
        year_filter=year_filter,
        use_cache=use_cache,
        cache_max_age_hours=cache_max_age_hours,
        ai_enhancement_level=enable_ai_enhancement,  # match parameter name
        gpt_rerank_top_n= 10,
        fetch_full_content=False,
        openai_api_key=api_key  # load from env in production
    )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

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
@router.get("/document/{doc_id}/summary")
async def get_document_summary(doc_id: str, force_refresh: bool = False):
    """
    Return a GPT summary for a document.

    - If cached and non-empty, returns the cached summary.
    - If empty or force_refresh=True, generates a new summary using GPT.
    - Falls back to GPT reasoning if summary is not generated.
    """
    logger.info(f"[SUMMARY] Start processing document {doc_id} | force_refresh={force_refresh}")

    # Load cache
    cache_data = load_cache()
    doc_entry = cache_data.get("documents", {}).get(doc_id)
    logger.info(f"[SUMMARY] Cache loaded, total documents in cache: {len(cache_data.get('documents', {}))}")

    # Check cached summary/reasoning
    cached_summary = None
    if doc_entry:
        cached_summary = doc_entry.get("gpt_summary") or doc_entry.get("gpt_reasoning")
    if cached_summary and not force_refresh:
        logger.info(f"[SUMMARY] Returning cached GPT summary/reasoning for doc {doc_id}")
        return {"summary": cached_summary, "from_cache": True}

    # Fetch full HTML content
    try:
        html_doc = await get_document_html(doc_id)

        # Build full_content from either content_text or sections
        full_content = html_doc.get("content_text", "")
        if not full_content.strip():
            sections = html_doc.get("sections", {})
            if sections and isinstance(sections, dict):
                full_content = "\n\n".join(
                    f"{title.upper()}\n{body}"
                    for title, body in sections.items()
                    if isinstance(body, str) and body.strip()
                )
        logger.info(f"[SUMMARY] Retrieved HTML content for doc {doc_id}, length={len(full_content)}")

        if not full_content.strip():
            logger.warning(f"[SUMMARY] Empty content for doc {doc_id}")
            raise HTTPException(status_code=404, detail="No content found to summarize")
    except Exception as e:
        logger.error(f"[SUMMARY] Failed to get HTML content for doc {doc_id}: {e}")
        raise

    # Prepare input for HybridEnhancer
    results = [{
        "title": html_doc.get("title", ""),
        "full_content": full_content,
        "content_source": "full_html"
    }]
    logger.info(f"[SUMMARY] Prepared input for HybridEnhancer for doc {doc_id}")

    # Call GPT to generate summary/reasoning
    try:
        enhancer = HybridEnhancer(api_key)
        enhanced_results, stats = await enhancer.enhance_results_hybrid(
            results=results,
            search_term=html_doc.get("title", ""),
            enhancement_level="gpt",
            gpt_rerank_top_n=1,
            fetch_full_content=False
        )
        logger.info(f"[SUMMARY] GPT enhancement completed for doc {doc_id}, stats: {stats}")
    except Exception as e:
        logger.error(f"[SUMMARY] GPT enhancement failed for doc {doc_id}: {e}")
        
        # ✅ Fallback: do only summarization
        enhanced_results = await summarize_only(results, html_doc.get("title", ""))
        stats = {"mode": "summary_only"}

    # Extract summary or reasoning
    first_result = enhanced_results[0] if isinstance(enhanced_results, list) else enhanced_results
    summary = first_result.get("gpt_summary") or first_result.get("gpt_reasoning") or "No summary generated"
    logger.info(f"[SUMMARY] Extracted summary/reasoning for doc {doc_id}, length={len(summary)}")

    # Save to cache
    documents = cache_data.setdefault("documents", {})
    if doc_id not in documents:
        documents[doc_id] = {"data": html_doc}
        logger.info(f"[SUMMARY] Added new document entry to cache for doc {doc_id}")
    documents[doc_id]["gpt_summary"] = first_result.get("gpt_summary")
    documents[doc_id]["gpt_reasoning"] = first_result.get("gpt_reasoning")
    save_cache(cache_data)
    logger.info(f"[SUMMARY] Saved GPT summary/reasoning to cache for doc {doc_id}")

    logger.info(f"[SUMMARY] Finished processing document {doc_id}")
    return {"summary": summary, "from_cache": False}


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
        relevance_score = traditional_relevance_calculator.calculate_relevance(search_term, case_text)
        
        # Extract facts section for additional analysis
        facts = traditional_fact_summarizer.extract_facts_section(case_text)
        
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
        relevance_score = traditional_relevance_calculator.calculate_relevance(test_search, test_text)
        
        # Test summarization
        summary = traditional_fact_summarizer.summarize_facts(test_text, test_search, 1)
        
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
def init_clients():
    """Initialize Qdrant and OpenAI clients with error handling."""
    try:
        qdrant = QdrantClient(
            host="localhost",  # Configure as needed
            port=6333,
            timeout=30.0
        )
        
        # Test connection
        qdrant.get_collections()
        logger.info("Qdrant client initialized successfully")
        
        # Initialize OpenAI client
        openai_client = AsyncOpenAI(api_key=api_key)  # Assumes API key is set in environment
        logger.info("OpenAI client initialized successfully")
        
        return qdrant, openai_client
        
    except Exception as e:
        logger.error(f"Failed to initialize clients: {str(e)}")
        raise

qdrant, openai_client = init_clients()

# Initialize tokenizer for token counting
try:
    tokenizer = tiktoken.encoding_for_model(config.CHAT_MODEL)
except KeyError:
    tokenizer = tiktoken.get_encoding("cl100k_base")


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

@router.post("/rag/document/index", response_model=Dict[str, Any])
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

@router.post("/rag/ask", response_model=RAGResponse)
async def ask_kenya_law(request: QueryRequest):
    """Production-grade RAG query endpoint."""
    start_time = datetime.now()
    
    logger.info(f"Received query: {request.query[:100]}...")
    
    try:
        # Generate query embedding using local model
        query_embedding = await embed_text(request.query)
        
        # Perform vector search
        try:
            search_results = qdrant.search(
                collection_name=config.COLLECTION_NAME,
                query_vector=query_embedding,
                limit=request.limit,
                score_threshold=request.min_score,
                with_payload=True
            )
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Vector search failed: {str(e)}"
            )
        
        logger.info(f"Vector search returned {len(search_results)} results")
        
        # Debug logging for empty results
        if len(search_results) == 0:
            logger.warning(f"No results found for query: '{request.query}' with min_score: {request.min_score}")
            
            # Try with lower threshold
            if request.min_score > 0.3:
                logger.info("Retrying search with lower threshold (0.2)")
                search_results = qdrant.search(
                    collection_name=config.COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=request.limit,
                    score_threshold=0.2,
                    with_payload=True
                )
                if search_results:
                    logger.info(f"Found {len(search_results)} results with lowered threshold")
        
        if not search_results:
            return RAGResponse(
                answer="I couldn't find any relevant Kenyan legal cases for your query. Please try rephrasing your question or using different legal terms.",
                sources=[],
                query=request.query,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        # Build context from search results
        context_parts = []
        sources = []
        
        for hit in search_results:
            payload = hit.payload
            
            context_part = f"""Case: {payload.get('title', 'Unknown')}
Court: {payload.get('court', 'Unknown')}
Date: {payload.get('date', 'Unknown')}
Citations: {', '.join(payload.get('citations', []))}

Content: {payload.get('snippet', '')}
"""
            context_parts.append(context_part)
            
            sources.append(Source(
                doc_id=payload.get('doc_id', ''),
                title=payload.get('title', 'Unknown Title'),
                court=payload.get('court'),
                date=payload.get('date'),
                citations=payload.get('citations', []),
                snippet=payload.get('snippet', ''),
                source_html_url=payload.get('source_html_url'),
                score=float(hit.score)
            ))
        
        context = "\n" + "="*80 + "\n".join(context_parts)
        
        # Generate response using OpenAI (only for chat completion)
        system_prompt = """You are an expert Kenyan legal assistant. Provide accurate, well-reasoned legal analysis based strictly on the provided case context.

Guidelines:
1. Only reference information from the provided cases
2. Cite specific cases when making legal points
3. Be precise about legal principles and precedents
4. If the context doesn't fully answer the question, say so
5. Use clear, professional legal language
6. Structure your response logically"""

        user_prompt = f"""Question: {request.query}

Legal Cases Context:
{context}

Please provide a comprehensive legal analysis based on the provided Kenyan cases."""

        completion = await openai_client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        answer = completion.choices[0].message.content
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=request.query,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"RAG query failed after {processing_time:.2f}s: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Query processing failed: {str(e)}"
        )

@router.get("/rag/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Qdrant connection
        qdrant.get_collections()
        
        # Test collection exists
        collection_info = qdrant.get_collection(config.COLLECTION_NAME)
        
        # Test embedding model
        test_embedding = await embed_text("test")
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "vector_size": len(test_embedding),
            "collection": {
                "name": config.COLLECTION_NAME,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.post("/admin/rag/recreate-collection")
async def recreate_collection():
    """Recreate collection with current model dimensions."""
    try:
        logger.info("Recreating collection...")
        
        # Delete existing collection if exists
        try:
            qdrant.delete_collection(config.COLLECTION_NAME)
            logger.info("Deleted existing collection")
        except Exception as e:
            logger.info(f"No existing collection to delete: {e}")
        
        # Create new collection with correct dimensions
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
        
        return {
            "status": "success",
            "message": "Collection recreated successfully",
            "collection_name": config.COLLECTION_NAME,
            "vector_size": config.VECTOR_SIZE,
            "embedding_model": config.EMBEDDING_MODEL_NAME
        }
        
    except Exception as e:
        logger.error(f"Failed to recreate collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/admin/rag/collection-info")
async def get_collection_info():
    """Get detailed collection information."""
    try:
        collection_info = qdrant.get_collection(config.COLLECTION_NAME)
        
        return {
            "collection_name": config.COLLECTION_NAME,
            "vector_size": collection_info.config.params.vectors.size,
            "expected_size": config.VECTOR_SIZE,
            "dimension_match": collection_info.config.params.vectors.size == config.VECTOR_SIZE,
            "distance_metric": collection_info.config.params.vectors.distance.value,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "current_embedding_model": config.EMBEDDING_MODEL_NAME,
            "status": collection_info.status
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "collection_exists": False,
            "expected_size": config.VECTOR_SIZE,
            "current_embedding_model": config.EMBEDDING_MODEL_NAME
        }

@router.get("/rag/models")
async def list_available_models():
    """List available embedding models."""
    return {
        "current_model": {
            "name": config.EMBEDDING_MODEL_NAME,
            "selected": config.SELECTED_MODEL,
            "vector_size": config.VECTOR_SIZE
        },
        "available_models": config.EMBEDDING_MODELS,
        "recommendations": {
            "fastest": "all-miniLM (384d)",
            "balanced": "bge-base (768d)",
            "highest_quality": "bge-large (1024d)",
            "best_for_legal": "bge-base or e5-base"
        }
    }

# Include your debug endpoints here as well...
@router.get("/rag/debug/collection")
async def debug_collection():
    """Debug endpoint to check collection contents."""
    try:
        collection_info = qdrant.get_collection(config.COLLECTION_NAME)
        
        # Get a few sample points to verify data structure
        sample_points = qdrant.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=3,
            with_payload=True,
            with_vectors=False
        )
        
        return {
            "collection_name": config.COLLECTION_NAME,
            "total_vectors": collection_info.vectors_count,
            "total_points": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "expected_size": config.VECTOR_SIZE,
            "dimension_match": collection_info.config.params.vectors.size == config.VECTOR_SIZE,
            "sample_points": [
                {
                    "id": point.id,
                    "payload_keys": list(point.payload.keys()) if point.payload else [],
                    "title": point.payload.get('title', 'N/A') if point.payload else 'N/A'
                }
                for point in sample_points[0]
            ] if sample_points[0] else []
        }
    except Exception as e:
        logger.error(f"Debug collection failed: {str(e)}")
        return {
            "error": str(e),
            "collection_exists": False,
            "expected_size": config.VECTOR_SIZE
        }