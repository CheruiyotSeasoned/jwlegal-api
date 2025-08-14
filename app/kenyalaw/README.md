# Kenya Law API Module

This module provides comprehensive access to Kenya Law documents with intelligent caching and advanced search capabilities.

## Features

- 🔍 **Advanced Search**: Search Kenya Law documents with filters
- 📦 **Intelligent Caching**: Single JSON file with document deduplication
- 🚀 **Fast API Integration**: RESTful endpoints with Pydantic validation
- 📊 **Cache Analytics**: Detailed statistics and management
- 🎯 **Document Lookup**: Find documents by ID or search cached content

## API Endpoints

### Search Documents

#### POST `/kenyalaw/search`
Search documents with full request body control.

**Request Body:**
```json
{
  "search_term": "murder",
  "page": 1,
  "page_size": 20,
  "court_filter": "High Court",
  "year_filter": "2020",
  "use_cache": true,
  "cache_max_age_hours": 24
}
```

#### GET `/kenyalaw/search`
Simple search with query parameters.

**Query Parameters:**
- `search_term` (required): Search keyword
- `page` (optional): Page number (default: 1)
- `page_size` (optional): Results per page (default: 20, max: 50)
- `court_filter` (optional): Filter by specific court
- `year_filter` (optional): Filter by specific year
- `use_cache` (optional): Enable caching (default: true)
- `cache_max_age_hours` (optional): Cache validity in hours (default: 24)

**Example:**
```bash
GET /kenyalaw/search?search_term=muruatetu&page=1&page_size=10
```

**Response:**
```json
{
  "results": [...],
  "total_count": 7058,
  "total_pages": 353,
  "current_page": 1,
  "facets": {...},
  "cached_at": "2024-01-15T10:30:00",
  "search_params": {...},
  "from_cache": false
}
```

### Document Management

#### GET `/kenyalaw/document/{document_id}`
Get a specific document by ID.

**Response:**
```json
{
  "document": {...},
  "from_cache": true,
  "source_searches": ["search_key1", "search_key2"]
}
```

### Cache Management

#### GET `/kenyalaw/cache/stats`
Get cache statistics.

**Response:**
```json
{
  "total_searches": 15,
  "total_documents": 250,
  "cache_size_mb": 2.5,
  "created_at": "2024-01-15T09:00:00",
  "last_updated": "2024-01-15T10:30:00"
}
```

#### DELETE `/kenyalaw/cache`
Clear all cached data.

#### GET `/kenyalaw/cache/search/{search_term}`
Search through cached documents.

**Response:**
```json
{
  "search_term": "murder",
  "total_found": 5,
  "documents": [
    {
      "doc_id": "123456",
      "document": {...},
      "match_type": "title",
      "source_searches": ["search_key1"]
    }
  ]
}
```

#### GET `/kenyalaw/cache/searches`
Get all cached searches.

**Response:**
```json
{
  "total_searches": 15,
  "searches": [
    {
      "search_key": "abc123...",
      "search_term": "murder",
      "page": 1,
      "total_results": 150,
      "cached_at": "2024-01-15T10:00:00"
    }
  ]
}
```

## Cache Structure

The caching system uses a single JSON file with intelligent deduplication:

```json
{
  "searches": {
    "search_key_hash": {
      "data": { /* complete search results */ },
      "cached_at": "timestamp",
      "search_params": { /* original parameters */ }
    }
  },
  "documents": {
    "document_id": {
      "data": { /* document data */ },
      "cached_at": "timestamp",
      "source_searches": ["search_key1", "search_key2"]
    }
  },
  "metadata": {
    "created_at": "timestamp",
    "last_updated": "timestamp",
    "total_searches": 0,
    "total_documents": 0
  }
}
```

## CLI Tool

Use the CLI tool for testing and development:

```bash
# Search documents
python kenyalaw_cli.py search "murder" --page 1 --page-size 10

# Search with filters
python kenyalaw_cli.py search "land dispute" --court "High Court" --year "2020"

# Get cache statistics
python kenyalaw_cli.py cache-stats

# Search cached documents
python kenyalaw_cli.py search-cached "muruatetu"

# Clear cache
python kenyalaw_cli.py clear-cache
```

## Key Benefits

1. **Efficient Storage**: Documents are stored only once, regardless of how many searches find them
2. **Fast Retrieval**: Single file cache with optimized structure
3. **Smart Relationships**: Tracks which searches found which documents
4. **Comprehensive API**: Full REST API with proper validation
5. **Easy Integration**: Seamlessly integrated into FastAPI application

## Usage in Main Application

The module is automatically included in the main FastAPI application. All endpoints are prefixed with `/kenyalaw/`.

Example usage:
```bash
curl "http://localhost:8000/kenyalaw/search?search_term=murder&page=1"
curl "http://localhost:8000/kenyalaw/cache/stats"
curl -X DELETE "http://localhost:8000/kenyalaw/cache"
```

## Error Handling

The API provides proper HTTP status codes and error messages:
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (document not found)
- `500`: Internal Server Error (API failures, network issues)

All errors include descriptive messages to help with debugging.


