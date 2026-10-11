"""
Receipt Agent - Processes documents and extracts expense information.

This agent:
- Extracts text from images/PDFs using OCR
- Parses structured fields using LLM
- Stores documents in database
- Matches with existing expenses (deduplication)
"""

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import Tool

from .base_agent import BaseAgent
from .tools.document_tools import create_document_tools


class ReceiptAgent(BaseAgent):
    """
    Specialized agent for receipt and document processing.
    
    Workflow:
    1. Receive document (image or PDF)
    2. Extract text using OCR
    3. Parse fields using LLM
    4. Check for duplicate expenses
    5. Store document with embeddings
    6. Return structured data
    """
    
    def __init__(
        self,
        region: str,
        llm: Any,
        database_url: str,
        embedding_model: Any
    ):
        """
        Initialize Receipt Agent.
        
        Args:
            region: AWS region where agent runs
            llm: LangChain LLM instance
            database_url: CockroachDB connection string
            embedding_model: Sentence transformer model
        """
        # Create document processing tools
        doc_tools = create_document_tools(database_url, embedding_model)
        
        # Add LLM-based parsing tool
        def parse_receipt_wrapper(text: str) -> str:
            """Wrapper to use agent's LLM for parsing"""
            return self._parse_receipt_fields(text)
        
        parse_tool = Tool(
            name="parse_receipt_fields",
            description="""Parse receipt text to extract structured fields using AI.
            Args: text (str) - raw text from receipt
            Returns: JSON with merchant, amount, date, category, items, payment_method""",
            func=parse_receipt_wrapper
        )
        
        all_tools = doc_tools + [parse_tool]
        
        # System prompt for Receipt Agent
        system_prompt = f"""You are a Receipt Processing Agent in region {region}.

Your job is to process receipts and invoices:
1. Extract text from images/PDFs using OCR tools
2. Parse the text to identify: merchant, amount, date, category, items
3. Check for duplicate expenses in the database
4. Store the processed document with embeddings for search

Always follow this workflow:
1. Use extract_text_from_image or extract_text_from_pdf
2. Use parse_receipt_fields to extract structured data
3. Use find_matching_expense to check for duplicates
4. Use store_document to save the processed receipt

Be thorough and accurate. If information is unclear, say so."""
        
        # Initialize base agent
        super().__init__(
            agent_type="receipt",
            region=region,
            llm=llm,
            tools=all_tools,
            database_url=database_url,
            system_prompt=system_prompt
        )
        
        self.embedding_model = embedding_model
    
    def _parse_receipt_fields(self, text: str) -> str:
        """
        Parse receipt text using LLM.
        
        Args:
            text: Raw text from OCR
        
        Returns:
            JSON string with extracted fields
        """
        try:
            prompt = f"""You are a JSON extraction bot. Your ONLY job is to output valid JSON, nothing else.

Extract these fields from the receipt text below and return ONLY the JSON object:

Receipt Text:
{text}

Required JSON format (do not include any text before or after the JSON):
{{
  "merchant": "store name",
  "amount": 0.00,
  "date": "YYYY-MM-DD",
  "category": "food or transportation or entertainment or shopping or services or other",
  "items": ["item1", "item2"],
  "payment_method": "credit card or debit card or cash or other"
}}

IMPORTANT: 
- Return ONLY the JSON object, no explanation, no text before or after
- If a field is not found, use null
- Amount must be a number, not a string
- Date must be in YYYY-MM-DD format

JSON:"""

            # Get LLM response
            response = self.llm.invoke(prompt)
            
            # Extract text from response (handle different LLM response formats)
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # Debug: Log the raw response
            print(f"🤖 LLM Raw Response: {response_text[:200]}...")
            
            # Clean up response (remove markdown code blocks if present)
            cleaned = response_text.strip()
            
            if not cleaned:
                raise ValueError("LLM returned empty response")
            
            # Remove markdown code blocks
            if cleaned.startswith('```json'):
                cleaned = cleaned.replace('```json', '').replace('```', '').strip()
            elif cleaned.startswith('```'):
                cleaned = cleaned.replace('```', '').strip()
            
            # Try to extract JSON if LLM wrapped it in text (common with some models)
            if not cleaned.startswith('{'):
                # Look for JSON object in the response
                first_brace = cleaned.find('{')
                if first_brace == -1:
                    print(f"⚠️  No JSON found in response: {cleaned[:500]}")
                    raise ValueError(f"LLM did not return valid JSON. Response: {cleaned[:200]}")
                
                # Try to find the complete JSON object by counting braces
                brace_count = 0
                in_string = False
                escape = False
                end_pos = first_brace
                
                for i in range(first_brace, len(cleaned)):
                    char = cleaned[i]
                    
                    # Handle string content (ignore braces inside strings)
                    if char == '"' and not escape:
                        in_string = not in_string
                    elif char == '\\' and in_string:
                        escape = not escape
                        continue
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    escape = False
                
                if brace_count == 0 and end_pos > first_brace:
                    cleaned = cleaned[first_brace:end_pos]
                    print(f"📝 Extracted JSON from text: {cleaned[:200]}...")
                else:
                    print(f"⚠️  Incomplete JSON in response: {cleaned[:500]}")
                    raise ValueError("Could not extract complete JSON object")
            
            # Try to parse as JSON
            print(f"📝 Attempting to parse JSON ({len(cleaned)} chars)...")
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
    
    def process_document(
        self,
        file_path: str,
        user_id: str,
        document_type: str = "receipt"
    ) -> dict[str, Any]:
        """
        Complete workflow: Process a document from start to finish.
        
        Args:
            file_path: Path to document file
            user_id: User who uploaded it
            document_type: Type of document (receipt, invoice, etc.)
        
        Returns:
            Dictionary with processing results
        """
        self.update_status("acting", {"action": "process_document", "file": file_path})
        
        result = {
            'success': False,
            'steps': [],
            'errors': []
        }
        
        try:
            # Step 1: Extract text
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                extract_result = self.execute_tool('extract_text_from_pdf', pdf_path=file_path)
            else:
                extract_result = self.execute_tool('extract_text_from_image', image_path=file_path)
            
            extract_data = json.loads(extract_result)
            result['steps'].append({
                'step': 'extract_text',
                'success': extract_data.get('success', False),
                'data': extract_data
            })
            
            if not extract_data.get('success'):
                result['errors'].append(f"Text extraction failed: {extract_data.get('error')}")
                self.update_status("idle")
                return result
            
            extracted_text = extract_data.get('text', '')
            
            # Step 2: Parse fields
            parse_result = self.execute_tool('parse_receipt_fields', text=extracted_text)
            parse_data = json.loads(parse_result)
            
            result['steps'].append({
                'step': 'parse_fields',
                'success': parse_data.get('success', False),
                'data': parse_data
            })
            
            if not parse_data.get('success'):
                result['errors'].append(f"Field parsing failed: {parse_data.get('error')}")
                self.update_status("idle")
                return result
            
            fields = parse_data.get('fields', {})
            
            # Step 3: Check for duplicates (if we have enough info)
            if fields.get('amount') and fields.get('date'):
                match_result = self.execute_tool(
                    'find_matching_expense',
                    amount=fields['amount'],
                    merchant=fields.get('merchant', ''),
                    date=fields['date'],
                    user_id=user_id
                )
                match_data = json.loads(match_result)
                
                result['steps'].append({
                    'step': 'find_duplicate',
                    'success': match_data.get('success', False),
                    'data': match_data
                })
                
                result['duplicate_check'] = {
                    'found': match_data.get('match_found', False),
                    'match': match_data.get('match'),
                    'candidates': match_data.get('candidates', [])
                }
            
            # Step 4: Store document
            store_result = self.execute_tool(
                'store_document',
                user_id=user_id,
                document_type=document_type,
                filename=Path(file_path).name,
                extracted_text=extracted_text,
                extracted_data=fields,
                s3_key=None  # Local processing for now
            )
            store_data = json.loads(store_result)
            
            result['steps'].append({
                'step': 'store_document',
                'success': store_data.get('success', False),
                'data': store_data
            })
            
            if not store_data.get('success'):
                result['errors'].append(f"Document storage failed: {store_data.get('error')}")
                self.update_status("idle")
                return result
            
            # Success!
            result['success'] = True
            result['document_id'] = store_data.get('document_id')
            result['extracted_fields'] = fields
            
            # Record decision
            self.store_decision(
                decision_type='document_processed',
                context={
                    'file': file_path,
                    'user_id': user_id,
                    'document_type': document_type
                },
                reasoning=f"Successfully processed {document_type} with {len(fields)} fields extracted",
                action={
                    'action': 'store_document',
                    'document_id': result['document_id']
                },
                confidence=0.9 if not result.get('duplicate_check', {}).get('found') else 0.7
            )
            
        except Exception as e:
            result['errors'].append(f"Unexpected error: {str(e)}")
        finally:
            self.update_status("idle")

        return result
    
    def process_batch(
        self,
        file_paths: list,
        user_id: str,
        document_type: str = "receipt"
    ) -> dict[str, Any]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths
            user_id: User who uploaded them
            document_type: Type of documents
        
        Returns:
            Dictionary with batch results
        """
        self.update_status("acting", {"action": "batch_process", "count": len(file_paths)})
        
        results = {
            'total': len(file_paths),
            'processed': 0,
            'failed': 0,
            'documents': []
        }
        
        for file_path in file_paths:
            result = self.process_document(file_path, user_id, document_type)
            
            if result['success']:
                results['processed'] += 1
            else:
                results['failed'] += 1
            
            results['documents'].append({
                'file': file_path,
                'result': result
            })
        
        self.update_status("idle")
        
        return results
