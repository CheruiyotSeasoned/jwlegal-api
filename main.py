from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
import os
import sys
from app.auth.routes import router as auth_router
from app.cases.routes import router as cases_router
from app.ocr.routes import router as ocr_router
from app.admin.routes import router as admin_router
from app.kenyalaw.routes import router as kenyalaw_router
from app.gpt.routes import router as gpt_router

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
    allow_origins=["http://localhost:8080", "http://192.168.1.135:8080",""],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(cases_router)
app.include_router(ocr_router)
app.include_router(admin_router)
app.include_router(kenyalaw_router)
app.include_router(gpt_router)
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
