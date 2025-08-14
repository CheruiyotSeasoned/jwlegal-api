from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import io
import os
from typing import List
from app.database import get_db
from app.models import User, AIUsageLog
from app.auth.dependencies import check_usage_limits

router = APIRouter(prefix="/ocr", tags=["OCR Document Processing"])

# Configure Tesseract path if needed (especially on Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@router.post("/upload")
def extract_text_from_document(
    file: UploadFile = File(...),
    current_user: User = Depends(check_usage_limits),
    db: Session = Depends(get_db)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check file type
    allowed_types = ['application/pdf', 'image/jpeg', 'image/png', 'image/tiff']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Please upload PDF, JPEG, PNG, or TIFF files."
        )
    
    try:
        extracted_text = ""
        file_content = file.file.read()
        
        if file.content_type == 'application/pdf':
            # Handle PDF with PyMuPDF
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # First try to extract text directly (for text-based PDFs)
                page_text = page.get_text()
                if page_text.strip():
                    extracted_text += page_text + "\n"
                else:
                    # If no text found, use OCR on the page image
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    ocr_text = pytesseract.image_to_string(image, lang='eng')
                    extracted_text += ocr_text + "\n"
            
            pdf_document.close()
            
        else:
            # Handle image files
            image = Image.open(io.BytesIO(file_content))
            extracted_text = pytesseract.image_to_string(image, lang='eng')
        
        # Clean up the extracted text
        extracted_text = extracted_text.strip()
        
        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the document"
            )
        
        # Update user usage
        current_user.monthly_requests_used += 1
        
        # Log usage
        usage_log = UsageLog(
            user_id=current_user.id,
            action="ocr_upload",
            cost_credits=1
        )
        db.add(usage_log)
        db.commit()
        
        return {
            "filename": file.filename,
            "extracted_text": extracted_text,
            "character_count": len(extracted_text),
            "word_count": len(extracted_text.split())
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
