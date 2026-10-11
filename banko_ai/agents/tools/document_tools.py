"""
Document processing tools for Receipt Agent.

Provides:
- OCR extraction from images/PDFs
- Text parsing and field extraction
- Document storage
"""

import json
import os
import tempfile
from datetime import datetime
from typing import Any

from langchain_core.tools import Tool
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

# OCR imports
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    from pypdf import PdfReader
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  OCR libraries not available. Install: pip install pytesseract pdf2image Pillow pypdf")


def create_document_tools(database_url: str, embedding_model) -> list[Tool]:
    """
    Create document processing tools.
    
    Args:
        database_url: CockroachDB connection string
        embedding_model: Sentence transformer for embeddings
    
    Returns:
        List of LangChain Tool objects
    """
    
    def extract_text_from_image(image_path: str) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to image file
        
        Returns:
            JSON string with extracted text
        """
        if not OCR_AVAILABLE:
            return json.dumps({
                'success': False,
                'error': 'OCR libraries not installed. Install: pip install pytesseract Pillow && sudo apt-get install tesseract-ocr'
            })
        
        try:
            # Open image
            image = Image.open(image_path)
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            
            # Get image info
            width, height = image.size
            format = image.format
            
            return json.dumps({
                'success': True,
                'text': text,
                'metadata': {
                    'width': width,
                    'height': height,
                    'format': format,
                    'text_length': len(text)
                }
            }, indent=2)
        
        except Exception as e:
            error_msg = str(e)
            # Add helpful hint for tesseract errors
            if 'tesseract' in error_msg.lower() or 'not installed' in error_msg.lower():
                error_msg = f"OCR failed: {error_msg}\n\n" \
                           f"💡 Solution: Install tesseract OCR:\n" \
                           f"   Ubuntu/Debian: sudo apt-get install -y tesseract-ocr\n" \
                           f"   macOS: brew install tesseract\n" \
                           f"   Or run: curl -fsSL https://raw.githubusercontent.com/cockroachlabs-field/banko-ai-assistant/main/scripts/setup_system_deps.sh | bash"
            
            return json.dumps({
                'success': False,
                'error': error_msg
            })
    
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from a PDF document.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            JSON string with extracted text
        """
        if not OCR_AVAILABLE:
            return json.dumps({
                'success': False,
                'error': 'PDF processing libraries not installed. Install: pip install pytesseract pdf2image Pillow pypdf && sudo apt-get install poppler-utils'
            })
        
        try:
            text_parts = []
            
            # Try direct text extraction first
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                for page_num in range(page_count):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(text)
            
            # If no text found, try OCR on images
            if not text_parts:
                images = convert_from_path(pdf_path)
                for i, image in enumerate(images):
                    text = pytesseract.image_to_string(image)
                    if text.strip():
                        text_parts.append(f"[Page {i+1}]\n{text}")
            
            full_text = "\n\n".join(text_parts)
            
            return json.dumps({
                'success': True,
                'text': full_text,
                'metadata': {
                    'pages': page_count if 'page_count' in locals() else len(text_parts),
                    'text_length': len(full_text),
                    'method': 'direct' if text_parts and 'images' not in locals() else 'ocr'
                }
            }, indent=2)
        
        except Exception as e:
            error_msg = str(e)
            # Add helpful hint for poppler errors
            if 'poppler' in error_msg.lower() or 'unable to get page count' in error_msg.lower():
                error_msg = f"Text extraction failed: {error_msg}\n\n" \
                           f"💡 Solution: Install poppler-utils system package:\n" \
                           f"   Ubuntu/Debian: sudo apt-get install -y poppler-utils\n" \
                           f"   macOS: brew install poppler\n" \
                           f"   Or run: curl -fsSL https://raw.githubusercontent.com/cockroachlabs-field/banko-ai-assistant/main/scripts/setup_system_deps.sh | bash"
            
            return json.dumps({
                'success': False,
                'error': error_msg
            })
    
    def parse_receipt_fields(text: str, llm: Any) -> str:
        """
        Parse receipt text to extract structured fields using LLM.
        
        Args:
            text: Raw text from receipt
            llm: LangChain LLM instance
        
        Returns:
            JSON string with extracted fields
        """
        try:
            prompt = f"""Extract the following fields from this receipt text:
            
Receipt Text:
{text}

Extract these fields (return as JSON):
- merchant: Name of the merchant/store
- amount: Total amount (number only, no currency symbol)
- date: Date of transaction (YYYY-MM-DD format if possible)
- category: Expense category (food, transportation, entertainment, shopping, services, other)
- items: List of purchased items (if visible)
- payment_method: How it was paid (credit card, debit card, cash, etc.)

Return ONLY valid JSON, no explanation. If a field is not found, use null.

Example:
{{
  "merchant": "Starbucks",
  "amount": 5.50,
  "date": "2024-11-04",
  "category": "food",
  "items": ["Coffee", "Muffin"],
  "payment_method": "credit card"
}}"""

            # Get LLM response
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse as JSON
            # Clean up response (remove markdown code blocks if present)
            cleaned = response_text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned.replace('```json', '').replace('```', '').strip()
            elif cleaned.startswith('```'):
                cleaned = cleaned.replace('```', '').strip()
            
            parsed = json.loads(cleaned)
            
            return json.dumps({
                'success': True,
                'fields': parsed,
                'raw_response': response_text
            }, indent=2)
        
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e),
                'raw_response': response_text if 'response_text' in locals() else None
            })
    
    def store_document(
        user_id: str,
        document_type: str,
        filename: str,
        extracted_text: str,
        extracted_data: dict[str, Any],
        s3_key: str | None = None
    ) -> str:
        """
        Store document information in the database.
        
        Args:
            user_id: User who uploaded the document
            document_type: Type of document (receipt, invoice, etc.)
            filename: Original filename
            extracted_text: Full extracted text
            extracted_data: Structured data extracted from document
            s3_key: S3 key if stored in object storage
        
        Returns:
            JSON string with document ID
        """
        try:
            # Generate embedding for searchability
            embedding = embedding_model.encode(extracted_text).tolist()
            
            # Format embedding as array literal for CockroachDB
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO documents
                    (user_id, document_type, s3_key, original_filename, 
                     extracted_text, extracted_data, embedding, 
                     processing_status, created_at)
                    VALUES
                    (:user_id, :doc_type, :s3_key, :filename,
                     :text, :data, CAST(:embedding AS VECTOR(384)),
                     'completed', :created)
                    RETURNING document_id
                """), {
                    'user_id': user_id,
                    'doc_type': document_type,
                    's3_key': s3_key or f'local/{filename}',
                    'filename': filename,
                    'text': extracted_text,
                    'data': json.dumps(extracted_data),
                    'embedding': embedding_str,
                    'created': datetime.utcnow()
                })
                
                document_id = result.scalar()
                conn.commit()
return json.dumps({
                'success': True,
                'document_id': str(document_id),
                'user_id': user_id,
                'filename': filename,
                'extracted_fields': extracted_data
            }, indent=2)
        
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            })
    
    def find_matching_expense(
        amount: float,
        merchant: str,
        date: str,
        user_id: str
    ) -> str:
        """
        Find existing expense that matches document data.
        
        Args:
            amount: Transaction amount
            merchant: Merchant name
            date: Transaction date (YYYY-MM-DD)
            user_id: User ID
        
        Returns:
            JSON string with matching expense or None
        """
        try:
            engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
            
            with engine.connect() as conn:
                # Look for exact match first
                result = conn.execute(text("""
                    SELECT 
                        expense_id, description, expense_amount, merchant, 
                        expense_date, shopping_type
                    FROM expenses
                    WHERE user_id = :user_id
                    AND expense_amount = :amount
                    AND merchant ILIKE :merchant
                    AND expense_date = :date::DATE
                    LIMIT 1
                """), {
                    'user_id': user_id,
                    'amount': amount,
                    'merchant': f'%{merchant}%',
                    'date': date
                })
                
                row = result.fetchone()
                
                if row:
                    match = {
                        'expense_id': str(row[0]),
                        'description': row[1],
                        'amount': float(row[2]),
                        'merchant': row[3],
                        'date': row[4].isoformat() if row[4] else None,
                        'category': row[5],
                        'match_type': 'exact'
                    }
                    
                    return json.dumps({
                        'success': True,
                        'match_found': True,
                        'match': match
                    }, indent=2)
                
                # Try fuzzy match on amount and date
                result = conn.execute(text("""
                    SELECT 
                        expense_id, description, expense_amount, merchant, 
                        expense_date, shopping_type,
                        ABS(expense_amount - :amount) as amount_diff
                    FROM expenses
                    WHERE user_id = :user_id
                    AND expense_date = :date::DATE
                    AND ABS(expense_amount - :amount) < 5.0
                    ORDER BY amount_diff
                    LIMIT 3
                """), {
                    'user_id': user_id,
                    'amount': amount,
                    'date': date
                })
                
                rows = result.fetchall()
                
                if rows:
                    candidates = []
                    for row in rows:
                        candidates.append({
                            'expense_id': str(row[0]),
                            'description': row[1],
                            'amount': float(row[2]),
                            'merchant': row[3],
                            'date': row[4].isoformat() if row[4] else None,
                            'category': row[5],
                            'amount_diff': float(row[6])
                        })
                    
                    return json.dumps({
                        'success': True,
                        'match_found': True,
                        'match_type': 'fuzzy',
                        'candidates': candidates
                    }, indent=2)
return json.dumps({
                'success': True,
                'match_found': False
            })
        
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            })
    
    # Create LangChain tools
    tools = [
        Tool(
            name="extract_text_from_image",
            description="""Extract text from an image file (JPG, PNG, etc.) using OCR.
            Args: image_path (str) - path to image file
            Returns: JSON with extracted text and metadata""",
            func=extract_text_from_image
        ),
        Tool(
            name="extract_text_from_pdf",
            description="""Extract text from a PDF document. Tries direct text extraction first, then OCR if needed.
            Args: pdf_path (str) - path to PDF file
            Returns: JSON with extracted text and metadata""",
            func=extract_text_from_pdf
        ),
        Tool(
            name="store_document",
            description="""Store processed document in database with embeddings for search.
            Args: user_id (str), document_type (str), filename (str), extracted_text (str), 
                  extracted_data (dict), s3_key (optional str)
            Returns: JSON with document_id""",
            func=lambda user_id, document_type, filename, extracted_text, extracted_data, s3_key=None:
                store_document(user_id, document_type, filename, extracted_text, extracted_data, s3_key)
        ),
        Tool(
            name="find_matching_expense",
            description="""Find existing expense that matches receipt data (for deduplication).
            Args: amount (float), merchant (str), date (str YYYY-MM-DD), user_id (str)
            Returns: JSON with matching expense or candidates""",
            func=lambda amount, merchant, date, user_id:
                find_matching_expense(amount, merchant, date, user_id)
        )
    ]
    
    # Note: parse_receipt_fields requires LLM, will be added by agent
    
    return tools
