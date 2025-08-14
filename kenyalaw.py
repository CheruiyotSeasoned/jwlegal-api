import requests
import time
import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional

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
                print(f"📦 Loaded cache: {len(cache_data.get('searches', {}))} searches, {len(cache_data.get('documents', {}))} documents")
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
        print(f"💾 Saved cache: {cache_data['metadata']['total_searches']} searches, {cache_data['metadata']['total_documents']} documents")
    except Exception as e:
        print(f"❌ Failed to save cache: {e}")

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

def clear_cache():
    """Clear the main cache file."""
    cache_filepath = get_cache_filepath()
    if os.path.exists(cache_filepath):
        os.remove(cache_filepath)
        print("🗑️  Cleared cache file")
    else:
        print("📁 Cache file does not exist.")

def get_cache_stats() -> Dict:
    """Get statistics about the cache."""
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
    
    return stats

def find_document_by_id(doc_id: str, cache_data: Dict) -> Optional[Dict]:
    """Find a document by ID in the cache."""
    return cache_data.get("documents", {}).get(str(doc_id), {}).get("data")

def search_cached_documents(search_term: str, cache_data: Dict) -> List[Dict]:
    """Search through cached documents by title, content, etc."""
    search_term_lower = search_term.lower()
    found_documents = []
    
    for doc_id, doc_info in cache_data.get("documents", {}).items():
        doc = doc_info.get("data", {})
        
        # Search in title
        title = doc.get("title", "").lower()
        if search_term_lower in title:
            found_documents.append({
                "doc_id": doc_id,
                "doc": doc,
                "match_type": "title",
                "source_searches": doc_info.get("source_searches", [])
            })
            continue
        
        # Search in citation
        citation = doc.get("citation", "").lower()
        if search_term_lower in citation:
            found_documents.append({
                "doc_id": doc_id,
                "doc": doc,
                "match_type": "citation",
                "source_searches": doc_info.get("source_searches", [])
            })
            continue
        
        # Search in case number
        case_numbers = doc.get("case_number", [])
        for case_num in case_numbers:
            if search_term_lower in case_num.lower():
                found_documents.append({
                    "doc_id": doc_id,
                    "doc": doc,
                    "match_type": "case_number",
                    "source_searches": doc_info.get("source_searches", [])
                })
                break
    
    return found_documents

def fetch_kenya_law_documents(
    search_term: str, 
    page: int = 1, 
    page_size: int = 20,
    max_results: int = 100,
    court_filter: Optional[str] = None,
    year_filter: Optional[str] = None,
    show_facets: bool = False,
    use_cache: bool = True,
    cache_max_age_hours: int = 24
) -> Dict:
    """
    Fetch Kenya law documents with pagination and filtering.
    
    Args:
        search_term: The search keyword
        page: Page number (default: 1)
        page_size: Number of results per page (default: 20, max: 50)
        max_results: Maximum total results to fetch (default: 100)
        court_filter: Filter by specific court
        year_filter: Filter by specific year
        show_facets: Whether to display facet information
        use_cache: Whether to use caching (default: True)
        cache_max_age_hours: Maximum age of cache in hours (default: 24)
    
    Returns:
        Dictionary containing results and metadata
    """
    
    # Check cache first if enabled
    if use_cache:
        cache_data = load_cache()
        search_key = get_search_key(search_term, page, page_size, court_filter, year_filter)
        
        # Check if we have cached search results
        cached_search = get_cached_search(search_key, cache_data)
        if cached_search and is_cache_valid(cache_data, cache_max_age_hours):
            print(f"📦 Using cached search results for: {search_term} (page {page})")
            return cached_search["data"]
    
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
        print(f"🌐 Fetching from API: {search_term} (page {page})")
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
            }
        }
        
        # Cache the response if caching is enabled
        if use_cache:
            cache_data = load_cache()
            search_key = get_search_key(search_term, page, page_size, court_filter, year_filter)
            cache_search_results(search_key, response_data, cache_data)
            save_cache(cache_data)
        
        return response_data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {"results": [], "total_count": 0, "total_pages": 0, "current_page": page}

def display_results(results_data: Dict, search_term: str):
    """Display search results in a formatted way."""
    results = results_data["results"]
    total_count = results_data["total_count"]
    current_page = results_data["current_page"]
    total_pages = results_data["total_pages"]
    
    print(f"\n🔍 Search Results for: '{search_term}'")
    print(f"📊 Total Results: {total_count:,}")
    print(f"📄 Page {current_page} of {total_pages}")
    print(f"📋 Showing {len(results)} results")
    print("=" * 80)
    
    if not results:
        print("❌ No results found.")
        return
    
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. 📄 {doc.get('title', 'No title')}")
        print(f"   📅 Date: {doc.get('date', 'No date')}")
        print(f"   🏛️  Court: {doc.get('court', 'No court')}")
        print(f"   🔗 URL: {doc.get('url', 'No URL')}")
        if doc.get('outcome'):
            print(f"   ⚖️  Outcome: {doc.get('outcome')}")
        print("-" * 80)

def display_facets(facets: Dict):
    """Display available facets for filtering."""
    print("\n🎯 Available Filters:")
    print("=" * 50)
    
    for facet_name, facet_data in facets.items():
        if facet_data and isinstance(facets[facet_name], dict):
            print(f"\n📂 {facet_name.upper()}:")
            for item in list(facets[facet_name].items())[:10]:  # Show first 10
                print(f"   • {item[0]}: {item[1]}")

def interactive_search():
    """Interactive search with pagination and filtering."""
    print("🏛️  Kenya Law Document Search")
    print("=" * 50)
    
    # Get search term
    search_term = input("\nEnter a legal keyword to search: ").strip()
    if not search_term:
        print("❌ Search term cannot be empty.")
        return
    
    # Get user preferences
    try:
        page_size = int(input("Results per page (10-50, default 20): ") or "20")
        page_size = max(10, min(50, page_size))
    except ValueError:
        page_size = 20
    
    try:
        max_results = int(input("Maximum total results (default 100): ") or "100")
        max_results = max(10, min(500, max_results))
    except ValueError:
        max_results = 100
    
    # Optional filters
    court_filter = input("Filter by court (optional, press Enter to skip): ").strip() or None
    year_filter = input("Filter by year (optional, press Enter to skip): ").strip() or None
    
    # Initial search
    current_page = 1
    total_fetched = 0
    
    while total_fetched < max_results:
        print(f"\n⏳ Fetching page {current_page}...")
        
        # Fetch results
        results_data = fetch_kenya_law_documents(
            search_term=search_term,
            page=current_page,
            page_size=page_size,
            court_filter=court_filter,
            year_filter=year_filter,
            show_facets=(current_page == 1)  # Show facets only on first page
        )
        
        if not results_data["results"]:
            if current_page == 1:
                print("❌ No results found for your search criteria.")
            break
        
        # Display results
        display_results(results_data, search_term)
        
        # Show facets on first page
        if current_page == 1 and results_data.get("facets"):
            display_facets(results_data["facets"])
        
        total_fetched += len(results_data["results"])
        
        # Check if we've reached the end
        if current_page >= results_data["total_pages"]:
            print(f"\n✅ Reached the end of results. Total fetched: {total_fetched}")
            break
        
        # Ask user if they want to continue
        if total_fetched < max_results:
            choice = input(f"\n📄 Continue to next page? (y/n, or 'q' to quit): ").lower().strip()
            if choice in ['n', 'q', 'no', 'quit']:
                print(f"✅ Search completed. Total results fetched: {total_fetched}")
                break
            current_page += 1
            time.sleep(1)  # Be nice to the API
        else:
            print(f"\n✅ Reached maximum results limit ({max_results})")
            break

def fetch_full_document(doc_id: str) -> Dict:
    """Fetch the full content of a specific document by ID."""
    url = f"https://new.kenyalaw.org/search/api/documents/{doc_id}/"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {}

def display_full_document(doc_data: Dict):
    """Display the complete document data structure."""
    print("\n" + "="*80)
    print("📄 FULL DOCUMENT RESPONSE STRUCTURE")
    print("="*80)
    
    import json
    print(json.dumps(doc_data, indent=2, ensure_ascii=False))
    
    print("\n" + "="*80)
    print("📋 KEY FIELDS SUMMARY")
    print("="*80)
    
    # Display key fields in a readable format
    key_fields = [
        'id', 'title', 'date', 'court', 'url', 'nature', 'outcome',
        'registry', 'locality', 'judges', 'authors', 'language',
        'labels', 'attorneys', 'matter_type', 'content', 'summary'
    ]
    
    for field in key_fields:
        if field in doc_data:
            value = doc_data[field]
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            print(f"🔹 {field}: {value}")
        else:
            print(f"🔹 {field}: Not available")

def display_search_result_structure(results_data: Dict):
    """Display the complete search results structure."""
    print("\n" + "="*80)
    print("📄 FULL SEARCH RESULTS RESPONSE STRUCTURE")
    print("="*80)
    
    import json
    print(json.dumps(results_data, indent=2, ensure_ascii=False))
    
    print("\n" + "="*80)
    print("📋 SEARCH RESPONSE SUMMARY")
    print("="*80)
    
    print(f"🔹 Total Count: {results_data.get('total_count', 'Not available')}")
    print(f"🔹 Total Pages: {results_data.get('total_pages', 'Not available')}")
    print(f"🔹 Current Page: {results_data.get('current_page', 'Not available')}")
    print(f"🔹 Results Count: {len(results_data.get('results', []))}")
    
    if results_data.get('facets'):
        print(f"🔹 Facets Available: {list(results_data['facets'].keys())}")
    
    # Show structure of first result if available
    if results_data.get('results'):
        first_result = results_data['results'][0]
        print(f"\n🔹 First Result Fields: {list(first_result.keys())}")
        
        print("\n📄 FIRST RESULT DETAILED VIEW:")
        print("-" * 50)
        for key, value in first_result.items():
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            print(f"🔹 {key}: {value}")

def batch_search(search_terms: List[str], output_file: str = "kenya_law_results.txt"):
    """Perform batch search for multiple terms and save to file."""
    print(f"🔄 Starting batch search for {len(search_terms)} terms...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Kenya Law Search Results\n")
        f.write("=" * 50 + "\n\n")
        
        for i, term in enumerate(search_terms, 1):
            print(f"\n🔍 Processing {i}/{len(search_terms)}: '{term}'")
            
            results_data = fetch_kenya_law_documents(
                search_term=term,
                page=1,
                page_size=10,  # Limit for batch processing
                max_results=10
            )
            
            f.write(f"\nSearch Term: {term}\n")
            f.write(f"Total Results: {results_data['total_count']}\n")
            f.write("-" * 40 + "\n")
            
            for doc in results_data["results"]:
                f.write(f"Title: {doc.get('title', 'No title')}\n")
                f.write(f"Date: {doc.get('date', 'No date')}\n")
                f.write(f"Court: {doc.get('court', 'No court')}\n")
                f.write(f"URL: {doc.get('url', 'No URL')}\n\n")
            
            time.sleep(2)  # Be nice to the API
    
    print(f"\n✅ Batch search completed. Results saved to: {output_file}")

def explore_document_structure():
    """Interactive function to explore document structure."""
    print("🔍 Document Structure Explorer")
    print("=" * 50)
    
    # First, get a search result to find a document ID
    search_term = input("Enter a search term to find documents: ").strip()
    if not search_term:
        print("❌ Search term cannot be empty.")
        return
    
    print(f"\n⏳ Searching for documents with term: '{search_term}'")
    results_data = fetch_kenya_law_documents(
        search_term=search_term,
        page=1,
        page_size=5,  # Just get a few results
        show_facets=True  # Show facets to see all available data
    )
    
    if not results_data["results"]:
        print("❌ No results found.")
        return
    
    # Display available documents
    print(f"\n📋 Found {len(results_data['results'])} documents:")
    for i, doc in enumerate(results_data["results"], 1):
        print(f"{i}. {doc.get('title', 'No title')} (ID: {doc.get('id', 'No ID')})")
    
    print(f"\n🔍 Options:")
    print("1. View complete search response structure")
    print("2. View individual document details (from search results)")
    print("3. Try to fetch full document (may not work)")
    
    try:
        choice = int(input(f"\nSelect option (1-3): "))
        
        if choice == 1:
            # Show the complete search response structure
            display_search_result_structure(results_data)
            
        elif choice == 2:
            # Let user choose which document to explore from search results
            doc_choice = int(input(f"\nSelect document to explore (1-{len(results_data['results'])}): "))
            if 1 <= doc_choice <= len(results_data["results"]):
                selected_doc = results_data["results"][doc_choice - 1]
                print(f"\n📄 DOCUMENT DETAILS FROM SEARCH RESULTS:")
                print("=" * 60)
                display_full_document(selected_doc)
            else:
                print("❌ Invalid selection.")
                
        elif choice == 3:
            # Try to fetch individual document (may fail)
            doc_choice = int(input(f"\nSelect document to fetch (1-{len(results_data['results'])}): "))
            if 1 <= doc_choice <= len(results_data["results"]):
                selected_doc = results_data["results"][doc_choice - 1]
                doc_id = selected_doc.get('id')
                
                if doc_id:
                    print(f"\n⏳ Attempting to fetch full document with ID: {doc_id}")
                    full_doc = fetch_full_document(doc_id)
                    
                    if full_doc:
                        display_full_document(full_doc)
                    else:
                        print("❌ Failed to fetch document details.")
                        print("📄 Showing document from search results instead:")
                        display_full_document(selected_doc)
                else:
                    print("❌ Document ID not available.")
            else:
                print("❌ Invalid selection.")
        else:
            print("❌ Invalid option selected.")
            
    except ValueError:
        print("❌ Please enter a valid number.")

def manage_cache():
    """Manage cache operations."""
    print("🗄️  Cache Management")
    print("=" * 50)
    print("1. View cache statistics")
    print("2. Clear all cache")
    print("3. Search cached documents")
    print("4. View cached searches")
    print("5. Find document by ID")
    print("6. Back to main menu")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == "1":
        stats = get_cache_stats()
        print(f"\n📊 Cache Statistics:")
        print(f"   Total searches: {stats['total_searches']}")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Cache size: {stats['cache_size_mb']:.2f} MB")
        print(f"   Created: {stats['created_at']}")
        print(f"   Last updated: {stats['last_updated']}")
    
    elif choice == "2":
        confirm = input("Are you sure you want to clear all cache? (y/n): ").lower().strip()
        if confirm == 'y':
            clear_cache()
        else:
            print("Cache clearing cancelled.")
    
    elif choice == "3":
        search_cached_documents_ui()
    
    elif choice == "4":
        view_cached_searches()
    
    elif choice == "5":
        find_document_by_id_ui()
    
    elif choice == "6":
        return
    
    else:
        print("❌ Invalid option selected.")

def search_cached_documents_ui():
    """Search through cached documents with UI."""
    print("🔍 Search Cached Documents")
    print("=" * 50)
    
    search_term = input("Enter search term to find in cached documents: ").strip()
    if not search_term:
        print("❌ Search term cannot be empty.")
        return
    
    cache_data = load_cache()
    found_documents = search_cached_documents(search_term, cache_data)
    
    if found_documents:
        print(f"\n📋 Found {len(found_documents)} cached documents:")
        for i, doc_info in enumerate(found_documents, 1):
            doc = doc_info["doc"]
            print(f"{i}. {doc.get('title', 'No title')}")
            print(f"   ID: {doc_info['doc_id']}")
            print(f"   Match: {doc_info['match_type']}")
            print(f"   Court: {doc.get('court', 'No court')}")
            print(f"   Date: {doc.get('date', 'No date')}")
            print(f"   Sources: {len(doc_info['source_searches'])} searches")
            print()
    else:
        print("❌ No cached documents found for that search term.")

def view_cached_searches():
    """View all cached searches."""
    print("📋 Cached Searches")
    print("=" * 50)
    
    cache_data = load_cache()
    searches = cache_data.get("searches", {})
    
    if not searches:
        print("❌ No cached searches found.")
        return
    
    print(f"\n📊 Found {len(searches)} cached searches:")
    for i, (search_key, search_info) in enumerate(searches.items(), 1):
        search_params = search_info.get("search_params", {})
        search_data = search_info.get("data", {})
        
        print(f"{i}. Search: {search_params.get('search_term', 'Unknown')}")
        print(f"   Page: {search_params.get('page', 1)}")
        print(f"   Results: {search_data.get('total_count', 0)}")
        print(f"   Cached: {search_info.get('cached_at', 'Unknown')}")
        print(f"   Key: {search_key[:16]}...")
        print()

def find_document_by_id_ui():
    """Find document by ID with UI."""
    print("🔍 Find Document by ID")
    print("=" * 50)
    
    doc_id = input("Enter document ID: ").strip()
    if not doc_id:
        print("❌ Document ID cannot be empty.")
        return
    
    cache_data = load_cache()
    doc = find_document_by_id(doc_id, cache_data)
    
    if doc:
        print(f"\n📄 Found document:")
        print(f"   ID: {doc_id}")
        print(f"   Title: {doc.get('title', 'No title')}")
        print(f"   Court: {doc.get('court', 'No court')}")
        print(f"   Date: {doc.get('date', 'No date')}")
        print(f"   Citation: {doc.get('citation', 'No citation')}")
        
        # Show source searches
        doc_info = cache_data.get("documents", {}).get(str(doc_id), {})
        source_searches = doc_info.get("source_searches", [])
        if source_searches:
            print(f"   Found in {len(source_searches)} searches")
    else:
        print("❌ Document not found in cache.")

if __name__ == "__main__":
    print("🏛️  Kenya Law Document Search Tool")
    print("=" * 50)
    print("1. Interactive Search")
    print("2. Batch Search")
    print("3. Quick Search (first 10 results)")
    print("4. Explore Document Structure (View Full API Response)")
    print("5. Cache Management")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        interactive_search()
    elif choice == "2":
        terms_input = input("Enter search terms separated by commas: ")
        search_terms = [term.strip() for term in terms_input.split(",") if term.strip()]
        if search_terms:
            batch_search(search_terms)
        else:
            print("❌ No valid search terms provided.")
    elif choice == "3":
        term = input("Enter a legal keyword: ").strip()
        if term:
            results_data = fetch_kenya_law_documents(term, page=1, page_size=10)
            display_results(results_data, term)
        else:
            print("❌ Search term cannot be empty.")
    elif choice == "4":
        explore_document_structure()
    elif choice == "5":
        manage_cache()
    else:
        print("❌ Invalid option selected.")
