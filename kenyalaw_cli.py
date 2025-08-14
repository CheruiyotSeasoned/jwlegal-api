#!/usr/bin/env python3
"""
Kenya Law CLI tool for testing the API integration
"""

import requests
import json
import argparse
from typing import Optional

BASE_URL = "http://localhost:8000"

def search_documents(
    search_term: str,
    page: int = 1,
    page_size: int = 20,
    court_filter: Optional[str] = None,
    year_filter: Optional[str] = None,
    use_cache: bool = True
):
    """Search documents using the API."""
    url = f"{BASE_URL}/kenyalaw/search"
    
    params = {
        "search_term": search_term,
        "page": page,
        "page_size": page_size,
        "use_cache": use_cache
    }
    
    if court_filter:
        params["court_filter"] = court_filter
    if year_filter:
        params["year_filter"] = year_filter
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"\n🔍 Search Results for: '{search_term}'")
        print(f"📊 Total Results: {data.get('total_count', 0):,}")
        print(f"📄 Page {data.get('current_page', 1)} of {data.get('total_pages', 0)}")
        print(f"📋 Showing {len(data.get('results', []))} results")
        print(f"📦 From Cache: {'Yes' if data.get('from_cache') else 'No'}")
        print("=" * 80)
        
        for i, doc in enumerate(data.get("results", []), 1):
            print(f"\n{i}. 📄 {doc.get('title', 'No title')}")
            print(f"   📅 Date: {doc.get('date', 'No date')}")
            print(f"   🏛️  Court: {doc.get('court', 'No court')}")
            print(f"   🔗 URL: {doc.get('url', 'No URL')}")
            print("-" * 80)
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def get_cache_stats():
    """Get cache statistics."""
    url = f"{BASE_URL}/kenyalaw/cache/stats"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"\n📊 Cache Statistics:")
        print(f"   Total searches: {data.get('total_searches', 0)}")
        print(f"   Total documents: {data.get('total_documents', 0)}")
        print(f"   Cache size: {data.get('cache_size_mb', 0):.2f} MB")
        print(f"   Created: {data.get('created_at', 'Unknown')}")
        print(f"   Last updated: {data.get('last_updated', 'Unknown')}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def clear_cache():
    """Clear cache."""
    url = f"{BASE_URL}/kenyalaw/cache"
    
    try:
        response = requests.delete(url)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ {data.get('message', 'Cache operation completed')}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def search_cached_documents(search_term: str):
    """Search cached documents."""
    url = f"{BASE_URL}/kenyalaw/cache/search/{search_term}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"\n🔍 Cached Document Search for: '{search_term}'")
        print(f"📋 Found {data.get('total_found', 0)} documents")
        print("=" * 80)
        
        for i, doc_info in enumerate(data.get("documents", []), 1):
            doc = doc_info["document"]
            print(f"\n{i}. 📄 {doc.get('title', 'No title')}")
            print(f"   ID: {doc_info['doc_id']}")
            print(f"   Match: {doc_info['match_type']}")
            print(f"   Court: {doc.get('court', 'No court')}")
            print(f"   Date: {doc.get('date', 'No date')}")
            print(f"   Sources: {len(doc_info['source_searches'])} searches")
            print("-" * 80)
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Kenya Law API CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("search_term", help="Search term")
    search_parser.add_argument("--page", type=int, default=1, help="Page number")
    search_parser.add_argument("--page-size", type=int, default=10, help="Results per page")
    search_parser.add_argument("--court", help="Filter by court")
    search_parser.add_argument("--year", help="Filter by year")
    search_parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    # Cache stats command
    subparsers.add_parser("cache-stats", help="Show cache statistics")
    
    # Clear cache command
    subparsers.add_parser("clear-cache", help="Clear all cache")
    
    # Search cached command
    cached_parser = subparsers.add_parser("search-cached", help="Search cached documents")
    cached_parser.add_argument("search_term", help="Search term")
    
    args = parser.parse_args()
    
    if args.command == "search":
        search_documents(
            search_term=args.search_term,
            page=args.page,
            page_size=args.page_size,
            court_filter=args.court,
            year_filter=args.year,
            use_cache=not args.no_cache
        )
    elif args.command == "cache-stats":
        get_cache_stats()
    elif args.command == "clear-cache":
        clear_cache()
    elif args.command == "search-cached":
        search_cached_documents(args.search_term)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
