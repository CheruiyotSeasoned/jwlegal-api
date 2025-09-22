from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.database import engine, Base
import os
import sys
from app.auth.routes import router as auth_router
from app.cases.routes import router as cases_router
from app.ocr.routes import router as ocr_router
from app.admin.routes import router as admin_router
from app.kenyalaw.routes import router as kenyalaw_router
from app.gpt.routes import router as gpt_router
from app.legalresearch.routes import router as legalresearch_router
from app.rag.routes import router as rag_router
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="JW Legal AI Backend",
    description="Kenyan-focused legal research and assistance platform",
    version="1.0.0"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://192.168.1.135:8080",
        "https://legalbuddy.aiota.online",  # ✅ only domain, no path
    ],
    allow_credentials=True,
    allow_methods=["*"],  # or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers=["*"],  # or ["Authorization", "Content-Type"]
)
#https://legalbuddyapi.aiota.online/auth/google-signup
#https://legalbuddyapi.aiota.online/auth/google-signup
#
# --- Add middleware for COOP/COEP headers ---
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups"
        response.headers["Cross-Origin-Embedder-Policy"] = "unsafe-none"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Include routers
app.include_router(auth_router)
app.include_router(cases_router)
app.include_router(ocr_router)
app.include_router(admin_router)
app.include_router(kenyalaw_router)
app.include_router(gpt_router)
app.include_router(legalresearch_router)
app.include_router(rag_router)
# Import and include new routers
try:
    from app.documents.routes import router as documents_router
    app.include_router(documents_router)
except ImportError as e:
    print(f"Warning: Could not import documents router: {e}")

try:
    from app.messaging.routes import router as messaging_router
    app.include_router(messaging_router)
except ImportError as e:
    print(f"Warning: Could not import messaging router: {e}")
@app.get("/")
def root():
    return {
        "message": "JW Legal AI Backend API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
