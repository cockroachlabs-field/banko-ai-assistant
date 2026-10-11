"""
Google Gemini AI provider implementation.

This module provides Google Vertex AI/Gemini integration for vector search and RAG responses.
"""

import json
import os
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError, OperationalError

from banko_ai.utils.embeddings import load_embedding_model

try:
    from google import genai
    from google.genai import types as genai_types
    from google.oauth2 import service_account
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ..utils.db_retry import TRANSIENT_ERRORS, create_resilient_engine, db_retry, get_database_url
from .base import AIAuthenticationError, AIConnectionError, AIProvider, RAGResponse, SearchResult
from .rag_prompts import build_banko_rag_prompt


class GeminiProvider(AIProvider):
    """Google Gemini AI provider implementation."""

    def __init__(self, config: dict[str, Any], cache_manager=None):
        """Initialize Gemini provider."""
        if not GEMINI_AVAILABLE:
            raise AIConnectionError(
                "Google GenAI SDK not available. Install with: pip install google-genai"
            )

        self.cache_manager = cache_manager

        self.project_id = config.get("project_id") or os.getenv("GOOGLE_PROJECT_ID")
        self.location = config.get("location") or os.getenv("GOOGLE_LOCATION", "us-central1")
        self.model_name = config.get("model") or os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-001")
        self.embedding_model = None
        self.db_engine = None
        self.genai_client = None
        self.credentials = None
        super().__init__(config)

    def _validate_config(self) -> None:
        """Validate Gemini configuration and initialize the google-genai client.

        Prefers Vertex AI auth (service account via GOOGLE_APPLICATION_CREDENTIALS)
        and falls back to the public Generative AI API if only GOOGLE_API_KEY is
        set. The google-genai SDK is a single client class for both backends —
        the `vertexai=True` flag switches it from public API to Vertex AI.

        Migrated 2026-05-22 off the vertexai SDK (deprecated 2025-06-24,
        removed 2026-06-24) per the upstream notice — same models, new SDK.
        """
        if not self.project_id:
            raise AIAuthenticationError("Google project ID is required")

        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        api_key = os.getenv("GOOGLE_API_KEY")

        try:
            if credentials_path and os.path.exists(credentials_path):
                # Vertex AI requires the cloud-platform OAuth scope explicitly.
                # Without it, genai.Client(vertexai=True,...) issues a token
                # the Vertex endpoint rejects as `invalid_scope`. The legacy
                # vertexai.init() path inferred this scope internally; the
                # google-genai client does not.
                self.credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                print(f"✅ Loaded Google credentials from: {credentials_path}")
                self.genai_client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location,
                    credentials=self.credentials,
                )
                print(
                    f"✅ Initialized google-genai (Vertex AI backend) with "
                    f"project: {self.project_id}, location: {self.location}, "
                    f"model: {self.model_name}"
                )
            elif api_key:
                self.genai_client = genai.Client(api_key=api_key)
                print(
                    f"✅ Initialized google-genai (Generative AI API) with "
                    f"model: {self.model_name}"
                )
            else:
                raise AIAuthenticationError(
                    "Google credentials are required: set "
                    "GOOGLE_APPLICATION_CREDENTIALS (service account JSON for "
                    "Vertex AI) or GOOGLE_API_KEY (Generative AI API)"
                )
        except Exception as e:
            if isinstance(e, AIConnectionError | AIAuthenticationError):
                raise
            raise AIConnectionError(f"Failed to initialize Gemini provider: {str(e)}")

    def get_default_model(self) -> str:
        """Get the default Gemini model."""
        return "gemini-1.5-pro"

    def get_available_models(self) -> list[str]:
        """Get available Gemini models. Override with GEMINI_MODELS env var or auto-discover from API."""
        extra = os.getenv("GEMINI_MODELS", "")
        if extra:
            return [m.strip() for m in extra.split(",") if m.strip()]
        
        if GEMINI_AVAILABLE:
            try:
                # Use existing client, or create one matching the auth method
                client = self.genai_client
                if not client:
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if api_key:
                        client = genai.Client(api_key=api_key)
                    elif self.project_id:
                        client = genai.Client(
                            vertexai=True,
                            project=self.project_id,
                            location=self.location,
                        )
                
                if client:
                    models = []
                    for m in client.models.list():
                        name = m.name.replace('models/', '') if m.name else ''
                        if 'gemini' in name:
                            supported = [a for a in (m.supported_actions or [])]
                            if 'generateContent' in supported:
                                models.append(name)
                    if models:
                        return sorted(models)
            except Exception as e:
                print(f"⚠️  Could not list Gemini models: {e}")
        
        return ["gemini-2.0-flash-001", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]

    def _get_embedding_model(self):
        """Get or create the embedding model."""
        if self.embedding_model is None:
            try:
                # Use configurable embedding model from environment or default
                embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                self.embedding_model = load_embedding_model(embedding_model_name)
            except Exception as e:
                raise AIConnectionError(f"Failed to load embedding model: {str(e)}")
        return self.embedding_model

    def _get_db_engine(self):
        """Get database engine with proper connection pooling."""
        if self.db_engine is None:
            database_url = get_database_url()
            try:
                self.db_engine = create_resilient_engine(database_url)
            except Exception as e:
                raise AIConnectionError(f"Failed to connect to database: {str(e)}")
        return self.db_engine

    @db_retry(max_attempts=3, initial_delay=0.5)
    def search_expenses(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
        threshold: float = 0.7
    ) -> list[SearchResult]:
        """Search for expenses using vector similarity with caching."""
        try:
            print("\n🔍 GEMINI VECTOR SEARCH (with caching):")
            print(f"1. Query: '{query}' | Limit: {limit}")
            
            # Generate query embedding with cache support
            if self.cache_manager:
                query_embedding = self.cache_manager._get_embedding_with_cache(query)
                print("2. Embedding generated (with cache support)")
                
                # Check vector_search_cache for cached results
                cached_results = self.cache_manager.get_cached_vector_search(query_embedding, limit)
                if cached_results:
                    print(f"3. ✅ Vector search cache HIT! Found {len(cached_results)} cached results")
                    # Convert cached dict results to SearchResult objects
                    results_list = []
                    for result in cached_results[:limit]:
                        results_list.append(SearchResult(
                            expense_id=result.get('expense_id', ''),
                            user_id=result.get('user_id', ''),
                            description=result.get('description', ''),
                            merchant=result.get('merchant', ''),
                            amount=result.get('expense_amount', 0),
                            date=result.get('expense_date', ''),
                            similarity_score=result.get('similarity_score', 0),
                            metadata={
                                'shopping_type': result.get('shopping_type', 'Unknown'),
                                'payment_method': result.get('payment_method', 'Unknown')
                            }
                        ))
                    return results_list
                print("3. ❌ Vector search cache MISS, querying database")
            else:
                embedding_model = self._get_embedding_model()
                query_embedding = embedding_model.encode([query])[0]
                print("2. Embedding generated (no cache available)")

            # Convert to PostgreSQL vector format (JSON string)
            search_embedding = json.dumps(query_embedding.tolist())

            # Build SQL query using named parameters (e.g., :search_embedding)
            sql = """
            SELECT
                expense_id,
                user_id,
                description,
                merchant,
                expense_amount,
                expense_date,
                embedding <=> :search_embedding as similarity_score,
                shopping_type,
                payment_method
            FROM expenses
            ORDER BY embedding <=> :search_embedding
            LIMIT :limit
            """

            # Build the parameters dictionary
            params = {
                "search_embedding": search_embedding,
                "threshold": threshold,
                "limit": limit
            }

            # Execute query using the dictionary of parameters
            engine = self._get_db_engine()
            with engine.connect() as conn:
                result = conn.execute(text(sql), params)
                rows = result.fetchall()

            print(f"4. Database returned {len(rows)} results")

            # Convert to SearchResult objects
            results = []
            for row in rows:
                results.append(SearchResult(
                    expense_id=str(row[0]),
                    user_id=str(row[1]),
                    description=row[2] or "",
                    merchant=row[3] or "",
                    amount=float(row[4]),
                    date=row[5].isoformat() if row[5] and hasattr(row[5], 'isoformat') else (str(row[5]) if row[5] else 'Unknown'),
                    similarity_score=float(row[6]),
                    metadata={
                        'shopping_type': row[7] or 'Unknown',
                        'payment_method': row[8] or 'Unknown'
                    }
                ))

            # Cache the results for future use
            if self.cache_manager and results:
                # Store as dict format for cache compatibility
                cache_results = []
                for r in results:
                    cache_results.append({
                        'expense_id': r.expense_id,
                        'user_id': r.user_id,
                        'description': r.description,
                        'merchant': r.merchant,
                        'expense_amount': r.amount,
                        'expense_date': r.date,
                        'similarity_score': r.similarity_score,
                        'shopping_type': r.metadata.get('shopping_type') if r.metadata else 'Unknown',
                        'payment_method': r.metadata.get('payment_method') if r.metadata else 'Unknown'
                    })
                self.cache_manager.cache_vector_search_results(query_embedding, cache_results)
                print(f"5. ✅ Cached {len(cache_results)} results in vector_search_cache")

            return results

        except TRANSIENT_ERRORS:
            # Let database errors bubble up so @db_retry decorator can handle them
            raise
        except Exception as e:
            raise AIConnectionError(f"Search failed: {str(e)}")

    def _get_financial_insights(self, search_results) -> dict:
        """Generate comprehensive financial insights from expense data (handles both SearchResult objects and dicts)."""
        if not search_results:
            return {}

        total_amount = 0
        categories = {}
        merchants = {}
        payment_methods = {}

        for result in search_results:
            # Handle both SearchResult objects and dictionaries
            if hasattr(result, 'amount'):
                # It's a SearchResult object
                amount = float(result.amount)
                merchant = result.merchant
                category = result.metadata.get('shopping_type', 'Unknown') if hasattr(result, 'metadata') and result.metadata else 'Unknown'
                payment = result.metadata.get('payment_method', 'Unknown') if hasattr(result, 'metadata') and result.metadata else 'Unknown'
            else:
                # It's a dictionary
                amount = float(result.get('expense_amount', 0))
                merchant = result.get('merchant', 'Unknown')
                category = result.get('shopping_type', 'Unknown')
                payment = result.get('payment_method', 'Unknown')

            total_amount += amount

            # Category analysis
            categories[category] = categories.get(category, 0) + amount

            # Merchant analysis
            merchants[merchant] = merchants.get(merchant, 0) + amount

            # Payment method analysis
            payment_methods[payment] = payment_methods.get(payment, 0) + amount

        # Find top categories and merchants
        top_category = max(categories.items(), key=lambda x: x[1]) if categories else None
        top_merchant = max(merchants.items(), key=lambda x: x[1]) if merchants else None

        return {
            'total_amount': total_amount,
            'num_transactions': len(search_results),
            'avg_transaction': total_amount / len(search_results) if search_results else 0,
            'categories': categories,
            'top_category': top_category,
            'top_merchant': top_merchant,
            'payment_methods': payment_methods
        }

    def _generate_budget_recommendations(self, insights: dict, prompt: str) -> str:
        """Generate personalized budget recommendations based on spending patterns (following WatsonX pattern)."""
        if not insights:
            return ""

        recommendations = []

        # High spending category recommendations
        if insights.get('top_category'):
            category, amount = insights['top_category']
            recommendations.append(f"Your highest spending category is **{category}** at **${amount:.2f}**. Consider setting a monthly budget limit for this category.")

        # Average transaction analysis
        avg_amount = insights.get('avg_transaction', 0)
        if avg_amount > 100:
            recommendations.append(f"Your average transaction is **${avg_amount:.2f}**. Consider reviewing larger purchases to identify potential savings.")

        # Merchant frequency analysis
        if insights.get('top_merchant'):
            merchant, amount = insights['top_merchant']
            recommendations.append(f"You frequently shop at **{merchant}** (${amount:.2f} total). Look for loyalty programs or discounts at this merchant.")

        # General budgeting tips
        if insights.get('total_amount', 0) > 500:
            recommendations.append("💡 **Budget Tip**: Consider the 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings and debt repayment.")

        return "\n".join(recommendations) if recommendations else ""

    def generate_rag_response(
        self,
        query: str,
        context: list[SearchResult],
        user_id: str | None = None,
        language: str = "en"
    ) -> RAGResponse:
        """Generate RAG response using Google Gemini (following WatsonX pattern)."""
        try:
            print("\n🤖 GOOGLE GEMINI RAG (with caching):")
            print(f"1. Query: '{query[:60]}...'")

            # DEBUG: Check what we received
            print(f"DEBUG: Context type: {type(context)}")
            print(f"DEBUG: Context length: {len(context)}")
            if context:
                print(f"DEBUG: First item type: {type(context[0])}")
                print(f"DEBUG: First item: {context[0]}")
                if hasattr(context[0], '__dict__'):
                    print(f"DEBUG: First item attributes: {context[0].__dict__}")

            # Check for cached response first
            if self.cache_manager:
                # Convert SearchResult objects to dict format for cache lookup
                search_results_dict = []
                for i, result in enumerate(context):
                    print(f"DEBUG: Processing context item {i}: type={type(result)}")
                    try:
                        # Handle both SearchResult objects and dictionaries
                        if hasattr(result, 'expense_id'):
                            # It's a SearchResult object
                            search_results_dict.append({
                                'expense_id': result.expense_id,
                                'user_id': result.user_id,
                                'description': result.description,
                                'merchant': result.merchant,
                                'expense_amount': result.amount,
                                'expense_date': result.date,
                                'similarity_score': result.similarity_score,
                                'shopping_type': result.metadata.get('shopping_type'),
                                'payment_method': result.metadata.get('payment_method'),
                                'recurring': result.metadata.get('recurring'),
                                'tags': result.metadata.get('tags')
                            })
                        else:
                            # It's already a dictionary - use it as is
                            search_results_dict.append(result)
                    except Exception as e:
                        print(f"DEBUG: Error processing context item {i}: {e}")
                        print(f"DEBUG: Item content: {result}")
                        raise

                cached_response = self.cache_manager.get_cached_response(
                    query, search_results_dict, "gemini", language=language
                )
                if cached_response:
                    print("2. ✅ Response cache HIT! Returning cached response")
                    return RAGResponse(
                        response=cached_response,
                        sources=context,
                        metadata={
                            'provider': 'gemini',
                            'model': self.get_default_model(),
                            'user_id': user_id,
                            'language': language,
                            'cached': True
                        }
                    )
                print("2. ❌ Response cache MISS, generating fresh response")
            else:
                print("2. No cache manager available, generating fresh response")

            # Generate financial insights from search results (following WatsonX pattern)
            print(f"DEBUG: About to call _get_financial_insights with {len(context)} items")
            try:
                insights = self._get_financial_insights(context)
                print(f"DEBUG: Financial insights generated successfully: {insights}")
            except Exception as e:
                print(f"DEBUG: Error in _get_financial_insights: {e}")
                insights = {}

            try:
                budget_recommendations = self._generate_budget_recommendations(insights, query)
                print("DEBUG: Budget recommendations generated successfully")
            except Exception as e:
                print(f"DEBUG: Error in _generate_budget_recommendations: {e}")
                budget_recommendations = ""

            # Prepare the search results context with enhanced analysis (handles both objects and dicts)
            search_results_text = ""
            if context:
                context_parts = []
                for result in context:
                    # Handle both SearchResult objects and dictionaries
                    if hasattr(result, 'amount'):
                        # It's a SearchResult object
                        description = result.description
                        merchant = result.merchant
                        amount = result.amount
                        date = result.date if hasattr(result, 'date') else 'Unknown'
                        shopping_type = result.metadata.get('shopping_type', 'Unknown') if hasattr(result, 'metadata') and result.metadata else 'Unknown'
                        payment_method = result.metadata.get('payment_method', 'Unknown') if hasattr(result, 'metadata') and result.metadata else 'Unknown'
                    else:
                        # It's a dictionary
                        description = result.get('description', '')
                        merchant = result.get('merchant', 'Unknown')
                        amount = result.get('expense_amount', 0)
                        
                        # Format date properly
                        date_value = result.get('expense_date')
                        if date_value and hasattr(date_value, 'isoformat'):
                            date = date_value.isoformat()
                        elif date_value and str(date_value) != 'None':
                            date = str(date_value)
                        else:
                            date = 'Unknown'
                        
                        shopping_type = result.get('shopping_type', 'Unknown')
                        payment_method = result.get('payment_method', 'Unknown')

                    context_parts.append(
                        f"• **{shopping_type}** at {merchant}: ${amount} on {date} ({payment_method}) - {description}"
                    )

                search_results_text = "\n".join(context_parts)

                # Add financial summary
                if insights:
                    search_results_text += "\n\n**📊 Financial Summary:**\n"
                    search_results_text += f"• Total Amount: **${insights['total_amount']:.2f}**\n"
                    search_results_text += f"• Number of Transactions: **{insights['num_transactions']}**\n"
                    search_results_text += f"• Average Transaction: **${insights['avg_transaction']:.2f}**\n"
                    if insights.get('top_category'):
                        cat, amt = insights['top_category']
                        search_results_text += f"• Top Category: **{cat}** (${amt:.2f})\n"
            else:
                search_results_text = "No specific expense records found for this query."

            enhanced_prompt = build_banko_rag_prompt(
                question=query,
                expense_data=search_results_text,
                budget_recommendations=budget_recommendations,
                language=language,
            )

            ai_response = ""
            try:
                if not self.genai_client:
                    raise AIConnectionError("Gemini client not initialized")

                response = self.genai_client.models.generate_content(
                    model=self.model_name,
                    contents=enhanced_prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=1000,
                        top_p=0.9,
                        top_k=40,
                    ),
                )

                if response and response.text:
                    ai_response = response.text
                else:
                    ai_response = "I apologize, but I couldn't generate a response at this time."

            except Exception as e:
                # Fallback to structured response if API call fails (following WatsonX pattern)
                print(f"⚠️ Gemini API call failed: {e}")
                default_recommendations = "• Monitor your spending patterns regularly\n• Consider setting up budget alerts\n• Review high-value transactions for optimization opportunities"
                ai_response = f"""## Financial Analysis for: "{query}"

### 📋 Transaction Details
{search_results_text}

### 📊 Financial Summary
${insights.get('total_amount', 0):.2f} total across {insights.get('num_transactions', 0)} transactions

### 🤖 AI-Powered Insights
Based on your expense data, I found {len(context)} relevant records. Here's a comprehensive analysis:

**Spending Analysis:**
- Total Amount: ${insights.get('total_amount', 0):.2f}
- Transaction Count: {insights.get('num_transactions', 0)}
- Average Transaction: ${insights.get('avg_transaction', 0):.2f}
- Top Category: {insights.get('top_category', ('Unknown', 0))[0] if insights.get('top_category') else 'Unknown'} (${insights.get('top_category', ('Unknown', 0))[1] if insights.get('top_category') else 0:.2f})

**Smart Recommendations:**
{budget_recommendations if budget_recommendations else default_recommendations}

**Next Steps:**
• Track your spending trends over time
• Set realistic budget goals for each category
• Review and optimize your payment methods

**Note**: API call failed, showing structured analysis above."""

            # Cache the response for future similar queries
            if self.cache_manager and ai_response:
                # Convert context items to dict format for caching (handle both objects and dicts)
                search_results_dict = []
                for result in context:
                    if hasattr(result, 'expense_id'):
                        # It's a SearchResult object
                        search_results_dict.append({
                            'expense_id': result.expense_id,
                            'user_id': result.user_id,
                            'description': result.description,
                            'merchant': result.merchant,
                            'expense_amount': result.amount,
                            'expense_date': result.date,
                            'similarity_score': result.similarity_score,
                            'shopping_type': result.metadata.get('shopping_type') if result.metadata else None,
                            'payment_method': result.metadata.get('payment_method') if result.metadata else None,
                            'recurring': result.metadata.get('recurring') if result.metadata else None,
                            'tags': result.metadata.get('tags') if result.metadata else None
                        })
                    else:
                        # It's already a dictionary - use it as is or ensure it has the right keys
                        search_results_dict.append({
                            'expense_id': result.get('expense_id', ''),
                            'user_id': result.get('user_id', ''),
                            'description': result.get('description', ''),
                            'merchant': result.get('merchant', ''),
                            'expense_amount': result.get('expense_amount', 0),
                            'expense_date': result.get('expense_date', ''),
                            'similarity_score': result.get('similarity_score', 0),
                            'shopping_type': result.get('shopping_type'),
                            'payment_method': result.get('payment_method'),
                            'recurring': result.get('recurring'),
                            'tags': result.get('tags')
                        })

                # Estimate token usage (rough approximation for Gemini)
                prompt_tokens = len(enhanced_prompt.split()) * 1.3  # ~1.3 tokens per word
                response_tokens = len(ai_response.split()) * 1.3

                if "API call failed" not in ai_response:
                    self.cache_manager.cache_response(
                        query, ai_response, search_results_dict, "gemini",
                        int(prompt_tokens), int(response_tokens),
                        language=language
                    )
                    print(f"3. ✅ Cached response (est. {int(prompt_tokens + response_tokens)} tokens)")
                else:
                    print("3. ⚠️ Skipped caching fallback response")

            return RAGResponse(
                response=ai_response,
                sources=context,
                metadata={
                    'provider': 'gemini',
                    "model": self.model_name,
                    "project_id": self.project_id,
                    "location": self.location,
                    "language": language,
                    'user_id': user_id,
                    'cached': False
                }
            )

        except Exception as e:
            # Return a structured error response like WatsonX does
            return RAGResponse(
                response=f"Sorry, I'm experiencing technical difficulties with Google Gemini. Error: {str(e)}",
                sources=context,
                metadata={
                    'provider': 'gemini',
                    'model': self.model_name,
                    'user_id': user_id,
                    'language': language,
                    'error': str(e)
                }
            )

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        try:
            embedding_model = self._get_embedding_model()
            embedding = embedding_model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            raise AIConnectionError(f"Embedding generation failed: {str(e)}")

    def test_connection(self) -> bool:
        """Test Gemini connection by listing models (no tokens consumed)."""
        try:
            if GEMINI_AVAILABLE and self.genai_client:
                models = list(self.genai_client.models.list())
                return len(models) > 0
            return False
        except Exception:
            return False
