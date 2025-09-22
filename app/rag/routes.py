from fastapi import APIRouter, HTTPException, Query
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import hashlib
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from fastapi.responses import StreamingResponse
import openai
from fastapi import FastAPI, File, UploadFile
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional, Any
import requests
import time
import json
from dotenv import load_dotenv
import re
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
import asyncio
import logging
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

router = APIRouter(prefix="/rag", tags=["Vector database"])

router = APIRouter()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333"))
)

# Ensure Qdrant collection exists
def init_collection():
    try:
        qdrant.get_collection("kenya_cases")
    except Exception:
        qdrant.recreate_collection(
            collection_name="kenya_cases",
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
init_collection()

def embed_text(text: str):
    """Generate embeddings with OpenAI."""
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def chunk_text(text, chunk_words=400, overlap=80):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_words - overlap):
        chunk = " ".join(words[i:i+chunk_words])
        chunks.append(chunk)
        if i + chunk_words >= len(words):
            break
    return chunks

def upsert_document(doc: dict):
    """Index Kenya Law case into Qdrant."""
    if not doc.get("content_text"):
        return

    chunks = chunk_text(doc["content_text"], chunk_words=400, overlap=80)
    points = []
    for i, chunk in enumerate(chunks):
        uid = hashlib.sha1(f"{doc.get('case_number')}_{i}".encode()).hexdigest()
        vector = embed_text(chunk)
        points.append(
            models.PointStruct(
                id=uid,
                vector=vector,
                payload={
                    "doc_id": doc.get("case_number"),
                    "title": doc.get("title"),
                    "date": doc.get("date"),
                    "court": doc.get("court"),
                    "citations": doc.get("citations"),
                    "snippet": chunk[:500],
                    "source_html_url": doc.get("source_html_url"),
                }
            )
        )
    qdrant.upsert(collection_name="kenya_cases", points=points)

@router.get("/document/html/{doc_id}")
async def get_document_html_and_index(doc_id: str):
    doc = await get_document_html(doc_id)  # re-use your existing function
    try:
        upsert_document(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    return {**doc, "indexed": True}


# ✅ New RAG query endpoint
@router.post("/ask")
async def ask_kenya_law(query: str = Query(..., description="Legal question")):
    try:
        q_emb = embed_text(query)
        hits = qdrant.search(
            collection_name="kenya_cases",
            query_vector=q_emb,
            limit=5,
            with_payload=True
        )

        if not hits:
            return {"answer": "No relevant cases found.", "sources": []}

        # Build context for LLM
        context = "\n---\n".join([
            f"[{h.payload['title']} - {h.payload['court']} - {h.payload['date']}]\n{h.payload['snippet']}"
            for h in hits
        ])

        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Kenyan legal assistant. Only use provided case context."},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
            ]
        )

        return {
            "answer": completion.choices[0].message.content,
            "sources": [h.payload for h in hits]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")
