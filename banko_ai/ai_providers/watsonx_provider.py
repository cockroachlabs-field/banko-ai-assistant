"""
IBM Watsonx AI Provider for Banko Assistant

This module provides integration with IBM Watsonx AI services to power the
Banko Assistant's conversational capabilities. It includes functions for:
- Expense data search and retrieval
- RAG (Retrieval Augmented Generation) responses
- Financial data analysis

Dependencies:
- requests: For HTTP API calls to Watsonx
- numpy: For vector operations
- json: For data serialization
- sentence_transformers: For embedding generation
- sqlalchemy: For database operations

Author: Banko AI Team
Date: 2025
"""

from typing import Any

import requests

from ..ai_providers.base import AIConnectionError, AIProvider, RAGResponse, SearchResult
from ..ai_providers.rag_prompts import build_banko_rag_prompt
from ..config.settings import get_config
from ..utils.db_retry import TRANSIENT_ERRORS, create_resilient_engine, db_retry, get_database_url


class WatsonxProvider(AIProvider):
    """IBM Watsonx AI Provider implementation."""
    
    def __init__(self, config: dict[str, Any] = None, cache_manager=None):
        """Initialize Watsonx provider with configuration."""
        if config is None:
            config = {}

        # Store config for base class compatibility
        self.config = config
        self.cache_manager = cache_manager

        cfg = get_config()
        self.api_key = config.get('api_key') or cfg.watsonx_api_key
        self.project_id = config.get('project_id') or cfg.watsonx_project_id
        self.current_model = config.get('model', config.get('model_id')) or cfg.watsonx_model

        # API configuration with environment variable support
        self.api_url = config.get('api_url') or cfg.watsonx_api_url
        self.token_url = config.get('token_url') or cfg.watsonx_token_url
        self.timeout = int(config.get('timeout', cfg.watsonx_timeout))
        self.embedding_model_name = config.get('embedding_model') or cfg.embedding_model
        
        # Discovery cache: get_available_models() can call IBM on every UI page
        # load; cache for the life of the process. App restart picks up changes.
        self._available_models_cache: list[str] | None = None

        # Make API key and project ID optional for demo purposes
        if not self.api_key:
            print("⚠️ WATSONX_API_KEY not found - running in demo mode")
        if not self.project_id:
            print("⚠️ WATSONX_PROJECT_ID not found - running in demo mode")
    
    def _validate_config(self) -> None:
        """Validate Watsonx configuration."""
        # Configuration is optional for demo mode
        pass
    
    def get_default_model(self) -> str:
        """Get the default model for Watsonx."""
        return 'openai/gpt-oss-120b'
    
    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for the given text."""
        try:
            from banko_ai.utils.embeddings import load_embedding_model
            model = load_embedding_model(self.embedding_model_name)
            embedding = model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
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
            print("\n🔍 WATSONX VECTOR SEARCH (with caching):")
            print(f"1. Query: '{query}' | Limit: {limit}")
            
            # Use the same simple search logic as the original watsonx.py
            import json

            from sqlalchemy import text

            from banko_ai.utils.embeddings import load_embedding_model
            
            # Database connection with proper pooling
            DB_URI = get_database_url()
            engine = create_resilient_engine(DB_URI)
            
            # Generate embedding with cache support
            if self.cache_manager:
                raw_embedding = self.cache_manager._get_embedding_with_cache(query)
                print("2. Embedding generated (with cache support)")
                
                # Check vector_search_cache for cached results
                cached_results = self.cache_manager.get_cached_vector_search(raw_embedding, limit, scope=str(user_id or ''))
                if cached_results:
                    print(f"3. ✅ Vector search cache HIT! Found {len(cached_results)} cached results")
                    # Convert cached dict results to SearchResult objects
                    results_list = []
                    for result in cached_results[:limit]:
                        # Format date properly
                        date_value = result.get('expense_date', '')
                        if date_value and hasattr(date_value, 'isoformat'):
                            date_str = date_value.isoformat()
                        elif date_value:
                            date_str = str(date_value)
                        else:
                            date_str = 'Unknown'
                        
                        results_list.append(SearchResult(
                            expense_id=result.get('expense_id', ''),
                            user_id=result.get('user_id', ''),
                            description=result.get('description', ''),
                            merchant=result.get('merchant', ''),
                            amount=result.get('expense_amount', 0),
                            date=date_str,
                            similarity_score=result.get('similarity_score', 0),
                            metadata={
                                'shopping_type': result.get('shopping_type', 'Unknown'),
                                'payment_method': result.get('payment_method', 'Unknown'),
                                'recurring': result.get('recurring', False),
                                'tags': result.get('tags', [])
                            }
                        ))
                    return results_list
                print("3. ❌ Vector search cache MISS, querying database")
            else:
                model = load_embedding_model(self.embedding_model_name)
                raw_embedding = model.encode(query)
                print("2. Embedding generated (no cache available)")
            
            search_embedding = json.dumps(raw_embedding.flatten().tolist())
            
            # Use complete query with all fields
            search_query = text("""
                SELECT 
                    expense_id,
                    user_id,
                    description,
                    merchant,
                    shopping_type,
                    expense_amount,
                    expense_date,
                    payment_method,
                    embedding <=> :search_embedding as similarity_score
                FROM expenses
                ORDER BY embedding <=> :search_embedding
                LIMIT :limit
            """)
            
            with engine.connect() as conn:
                results = conn.execute(search_query, 
                                     {'search_embedding': search_embedding, 'limit': limit})
                search_results = [dict(row._mapping) for row in results]
                print(f"4. Database returned {len(search_results)} results")
                
                # Convert to SearchResult objects
                results_list = []
                for result in search_results:
                    # Format date properly
                    date_value = result.get('expense_date', '')
                    if date_value and hasattr(date_value, 'isoformat'):
                        date_str = date_value.isoformat()
                    elif date_value:
                        date_str = str(date_value)
                    else:
                        date_str = 'Unknown'
                    
                    results_list.append(SearchResult(
                        expense_id=result.get('expense_id', ''),
                        user_id=result.get('user_id', ''),
                        description=result['description'],
                        merchant=result['merchant'],
                        amount=result['expense_amount'],
                        date=date_str,
                        similarity_score=result['similarity_score'],
                        metadata={
                            'shopping_type': result.get('shopping_type', 'Unknown'),
                            'payment_method': result.get('payment_method', 'Unknown'),
                            'recurring': False,
                            'tags': []
                        }
                    ))
                
                # Cache the results for future use
                if self.cache_manager and search_results:
                    # Store as dict format for cache compatibility
                    cache_results = []
                    for result in search_results:
                        # Convert date to string if it's a date object
                        expense_date = result.get('expense_date', '')
                        if hasattr(expense_date, 'isoformat'):
                            expense_date = expense_date.isoformat()
                        
                        cache_results.append({
                            'expense_id': result.get('expense_id', ''),
                            'user_id': result.get('user_id', ''),
                            'description': result['description'],
                            'merchant': result['merchant'],
                            'expense_amount': result['expense_amount'],
                            'expense_date': str(expense_date) if expense_date else '',
                            'similarity_score': result['similarity_score'],
                            'shopping_type': result.get('shopping_type', 'Unknown'),
                            'payment_method': result.get('payment_method', 'Unknown'),
                            'recurring': False,
                            'tags': []
                        })
                    self.cache_manager.cache_vector_search_results(raw_embedding, cache_results, scope=str(user_id or ''))
                    print(f"5. ✅ Cached {len(cache_results)} results in vector_search_cache")
                
                return results_list
                
        except TRANSIENT_ERRORS:
            # Let database errors bubble up so @db_retry decorator can handle them
            raise
        except Exception as e:
            print(f"Error in search_expenses: {e}")
            raise AIConnectionError(f"Search failed: {str(e)}")
    
    def get_available_models(self) -> list[str]:
        """Get list of available Watsonx models.

        Order of preference:
        1. `WATSONX_MODELS` env var (comma-separated) — for airgap/pinned
           environments; read fresh each call so env changes take effect.
        2. Cached result from a prior IBM `foundation_model_specs` discovery.
        3. A fresh discovery call, cached on success.
        4. A static fallback stub of known-good chat models, so the dropdown
           still has multiple recoverable options when discovery fails.
        """
        cfg = get_config()
        extra = cfg.watsonx_models
        if extra:
            return [m.strip() for m in extra.split(",") if m.strip()]

        if self._available_models_cache is not None:
            return self._available_models_cache

        try:
            access_token = self._get_access_token()
            api_url = cfg.watsonx_base_url
            if '/ml/v1' in api_url or '?' in api_url:
                from urllib.parse import urlparse
                parsed = urlparse(api_url)
                api_url = f"{parsed.scheme}://{parsed.netloc}"

            resp = requests.get(
                f"{api_url}/ml/v1/foundation_model_specs?version=2023-05-29&limit=200",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                models = []
                for m in resp.json().get('resources', []):
                    tasks = [t.get('id', '') for t in m.get('tasks', [])]
                    if 'generation' in tasks or 'chat' in tasks:
                        models.append(m['model_id'])
                if models:
                    self._available_models_cache = sorted(models)
                    return self._available_models_cache
        except Exception as e:
            print(f"⚠️  Could not list Watsonx models: {e}")

        # Fallback: a small set of known-good instruction-tuned models. Code-
        # tuned models are deliberately excluded — they echo prompt templates
        # back on JSON extraction tasks (caught by ReceiptExtraction validation
        # but bad UX). Not cached so transient IBM outages let discovery
        # recover on the next call.
        return [
            "meta-llama/llama-3-3-70b-instruct",
            "mistralai/mistral-large",
            "ibm/granite-3-8b-instruct",
            "openai/gpt-oss-120b",
        ]
    
    def set_model(self, model_id: str) -> bool:
        """Set the current model."""
        if model_id in self.get_available_models():
            self.current_model = model_id
            return True
        return False
    
    def test_connection(self) -> bool:
        """Test connection to Watsonx API."""
        if not self.api_key or not self.project_id:
            return False
        
        try:
            # Just test if we can get an access token - this is faster and more reliable
            # than making a full API call for status checks
            access_token = self._get_access_token()
            return access_token is not None
        except Exception:
            # If token request fails, the connection is not available
            return False
    
    def _get_access_token(self):
        """Get IBM Cloud access token from API key (copied from original)."""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.api_key
        }
        
        try:
            response = requests.post(self.token_url, headers=headers, data=data, timeout=self.timeout)
            if response.status_code != 200:
                raise Exception(f"Failed to get access token (status {response.status_code}): {response.text}")
            token_data = response.json()
            access_token = token_data.get('access_token')
            if access_token:
                return access_token
            else:
                raise Exception("No access token in response")
        except Exception as e:
            raise Exception(f"Token request failed: {str(e)}")
    
    def _call_watsonx_api(self, messages):
        """Make a direct API call to IBM Watsonx chat endpoint (copied from original)."""
        access_token = self._get_access_token()
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        body = {
            "project_id": self.project_id,
            "model_id": self.current_model,
            "messages": messages,
            "frequency_penalty": 0,
            "max_tokens": 2000,
            "presence_penalty": 0,
            "temperature": 0.7,
            "top_p": 1
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=body,
                timeout=self.timeout
            )
            if response.status_code != 200:
                raise Exception(f"Watsonx API error (status {response.status_code}): {response.text}")
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content']
            elif 'generated_text' in data:
                return data['generated_text']
            else:
                print(f"Unexpected Watsonx response format: {data}")
                return "I apologize, but I'm having trouble generating a response right now."
        except Exception as e:
            raise Exception(f"Watsonx API call failed: {str(e)}")
    
    def _get_financial_insights(self, search_results: list[SearchResult]) -> dict:
        """Generate comprehensive financial insights from expense data (copied from original)."""
        if not search_results:
            return {}
        
        total_amount = sum(float(result.amount) for result in search_results)
        categories = {}
        merchants = {}
        payment_methods = {}
        
        for result in search_results:
            # Category analysis
            category = result.metadata.get('shopping_type', 'Unknown')
            categories[category] = categories.get(category, 0) + float(result.amount)
            
            # Merchant analysis
            merchant = result.merchant
            merchants[merchant] = merchants.get(merchant, 0) + float(result.amount)
            
            # Payment method analysis
            payment = result.metadata.get('payment_method', 'Unknown')
            payment_methods[payment] = payment_methods.get(payment, 0) + float(result.amount)
        
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
        """Generate personalized budget recommendations based on spending patterns (copied from original)."""
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
    
    def simple_rag_response(self, prompt: str, search_results: list[dict[str, Any]], language: str = "English") -> str:
        """
        Simple RAG response that matches the original implementation exactly.
        Takes a prompt and list of dictionaries (like original search results).
        """
        try:
            print("\n🤖 SIMPLE WATSONX RAG:")
            print(f"1. Query: '{prompt[:60]}...'")
            
            # Check for cached response first
            if self.cache_manager:
                cached_response = self.cache_manager.get_cached_response(
                    prompt, search_results, "watsonx", language=language
                )
                if cached_response:
                    print("2. ✅ Response cache HIT! Returning cached response")
                    return cached_response
                print("2. ❌ Response cache MISS, generating fresh response")
            else:
                print("2. No cache manager available, generating fresh response")
            
            # Generate financial insights and categorization analysis (matching original)
            insights = self._get_financial_insights_from_dicts(search_results)
            budget_recommendations = self._generate_budget_recommendations(insights, prompt)
            
            # Prepare the search results context with enhanced analysis (matching original)
            search_results_text = ""
            if search_results:
                search_results_text = "\n".join(
                    f"• **{result.get('shopping_type', 'Unknown')}** at {result['merchant']}: ${result['expense_amount']} on {result.get('expense_date', 'Unknown')} ({result.get('payment_method', 'Unknown')}) - {result['description']}"
                    for result in search_results
                )
                
                # Add financial summary (matching original)
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
                question=prompt,
                expense_data=search_results_text,
                budget_recommendations=budget_recommendations,
                language=language,
            )
            
            # Prepare messages for chat format (matching original)
            messages = [
                {
                    "role": "user",
                    "content": enhanced_prompt
                }
            ]
            
            # Call Watsonx API (matching original implementation)
            print("3. 🔄 Calling Watsonx API...")
            response = self._call_watsonx_api(messages)
            print("4. ✅ Watsonx response generated successfully")
            
            # Cache the response for future similar queries
            if self.cache_manager and response:
                # Estimate token usage (rough approximation)
                prompt_tokens = len(enhanced_prompt.split()) * 1.3  # ~1.3 tokens per word
                response_tokens = len(response.split()) * 1.3
                
                if "API call failed" not in response:
                    self.cache_manager.cache_response(
                        prompt, response, search_results, "watsonx",
                        int(prompt_tokens), int(response_tokens),
                        language=language
                    )
                    print(f"5. ✅ Cached response (est. {int(prompt_tokens + response_tokens)} tokens)")
                else:
                    print("5. ⚠️ Skipped caching fallback response")
            
            return response
            
        except Exception as e:
            error_msg = f"❌ Error generating Watsonx response: {str(e)}"
            print(error_msg)

            # Check if it's a network connectivity issue
            from .display import NEUTRAL_UNAVAILABLE
            if "Failed to resolve" in str(e) or "nodename nor servname provided" in str(e) or "Network connectivity issue" in str(e):
                print(f"Network connectivity issues with IBM Watsonx AI: {e}")
                return NEUTRAL_UNAVAILABLE
            else:
                print(f"Technical difficulties with IBM Watsonx AI: {e}")
                return NEUTRAL_UNAVAILABLE
    
    def _get_financial_insights_from_dicts(self, search_results: list[dict[str, Any]]) -> dict:
        """Generate financial insights from dictionary format (matching original)."""
        if not search_results:
            return {}
        
        total_amount = sum(float(result['expense_amount']) for result in search_results)
        categories = {}
        merchants = {}
        
        for result in search_results:
            # Category analysis
            category = result['shopping_type']
            categories[category] = categories.get(category, 0) + float(result['expense_amount'])
            
            # Merchant analysis
            merchant = result['merchant']
            merchants[merchant] = merchants.get(merchant, 0) + float(result['expense_amount'])
        
        # Find top categories and merchants
        top_category = max(categories.items(), key=lambda x: x[1]) if categories else None
        top_merchant = max(merchants.items(), key=lambda x: x[1]) if merchants else None
        
        return {
            'total_amount': total_amount,
            'num_transactions': len(search_results),
            'avg_transaction': total_amount / len(search_results) if search_results else 0,
            'categories': categories,
            'top_category': top_category,
            'top_merchant': top_merchant
        }

    def rag_response(
        self,
        query: str,
        context: list[SearchResult],
        language: str = "en"
    ) -> str:
        """Generate a RAG response using Watsonx API (copied from original working implementation)."""
        try:
            if not self.api_key or not self.project_id:
                # Return structured demo response if no API credentials
                if not context:
                    return f"""## Financial Analysis for: "{query}"

### 📋 Transaction Details
No expense records found for this query.

### 📊 Financial Summary
No data available for analysis.

### 🤖 AI-Powered Insights
I couldn't find any relevant expense records for your query. Please try:
- Different keywords (e.g., "groceries", "restaurants", "transportation")
- Broader categories (e.g., "food", "shopping", "bills")
- Time periods (e.g., "last month", "this week")

**Note**: I need API credentials to generate more detailed AI-powered insights."""

                # Generate financial insights from search results
                insights = self._get_financial_insights(context)
                budget_recommendations = self._generate_budget_recommendations(insights, query)
                
                # Create table text from search results
                table_text = ""
                if context:
                    table_text = "\n".join([
                        f"• **{result.metadata.get('shopping_type', 'Unknown')}** at {result.merchant}: ${result.amount} ({result.metadata.get('payment_method', 'Unknown')}) - {result.description}"
                        for result in context
                    ])
                
                # Create context text with financial summary
                context_text = f"""**📊 Financial Summary:**
• Total Amount: ${insights.get('total_amount', 0):.2f}
• Number of Transactions: {insights.get('num_transactions', 0)}
• Average Transaction: ${insights.get('avg_transaction', 0):.2f}
• Top Category: {insights.get('top_category', ('Unknown', 0))[0] if insights.get('top_category') else 'Unknown'}
• Most frequent category: {insights.get('top_category', ('Unknown', 0))[0] if insights.get('top_category') else 'Unknown'}

**Recommendations:**
{budget_recommendations if budget_recommendations else '• Consider reviewing your spending patterns regularly' + chr(10) + '• Set up budget alerts for high-value categories'}

**Note**: I can see {len(context)} relevant expense records, but I need API credentials to generate more detailed AI-powered insights."""
            else:
                # Make actual Watsonx API call with enhanced prompt (copied from original)
                try:
                    # Generate financial insights from search results
                    insights = self._get_financial_insights(context)
                    budget_recommendations = self._generate_budget_recommendations(insights, query)
                    
                    # Prepare the search results context with enhanced analysis (copied from original)
                    search_results_text = ""
                    if context:
                        search_results_text = "\n".join(
                            f"• **{result.metadata.get('shopping_type', 'Unknown')}** at {result.merchant}: ${result.amount} ({result.metadata.get('payment_method', 'Unknown')}) - {result.description}"
                            for result in context
                        )
                        
                        # Add financial summary (copied from original)
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

                    messages = [
                        {
                            "role": "user",
                            "content": enhanced_prompt
                        }
                    ]

                    # Call Watsonx API
                    response_text = self._call_watsonx_api(messages)
                    
                except Exception as e:
                    # Fallback to structured response if API call fails
                    error_msg = str(e)
                    response_text = "## Financial Analysis for: \"" + query + "\"\n\n"
                    response_text += "### 📋 Transaction Details\n"
                    response_text += search_results_text if 'search_results_text' in locals() else 'No data available'
                    response_text += "\n\n### 📊 Financial Summary\n"
                    response_text += f"{insights.get('total_amount', 0):.2f} total across {insights.get('num_transactions', 0)} transactions"
                    response_text += "\n\n### 🤖 AI-Powered Insights\n"
                    response_text += f"Based on your expense data, I found {len(context)} relevant records. Here's a comprehensive analysis:\n\n"
                    response_text += "**Spending Analysis:**\n"
                    response_text += f"- Total Amount: ${insights.get('total_amount', 0):.2f}\n"
                    response_text += f"- Transaction Count: {insights.get('num_transactions', 0)}\n"
                    response_text += f"- Average Transaction: ${insights.get('avg_transaction', 0):.2f}\n"
                    top_category = insights.get('top_category', ('Unknown', 0))
                    response_text += "- Top Category: " + (top_category[0] if top_category else 'Unknown') + " ($" + f"{top_category[1]:.2f}" + " if top_category else 0)\n\n"
                    response_text += "**Smart Recommendations:**\n"
                    response_text += budget_recommendations if budget_recommendations else '• Monitor your spending patterns regularly\n• Consider setting up budget alerts\n• Review high-value transactions for optimization opportunities'
                    response_text += "\n\n**Next Steps:**\n"
                    response_text += "• Track your spending trends over time\n"
                    response_text += "• Set realistic budget goals for each category\n"
                    response_text += "• Review and optimize your payment methods\n\n"
                    response_text += "**Note**: API call failed, showing structured analysis above."
            
            return response_text

        except Exception as e:
            from .display import NEUTRAL_UNAVAILABLE
            print(f"Sorry, I'm experiencing technical difficulties. Error: {str(e)}")
            return NEUTRAL_UNAVAILABLE
    
    def generate_rag_response(
        self,
        query: str,
        context: list[SearchResult],
        user_id: str | None = None,
        language: str = "en"
    ) -> RAGResponse:
        """Generate a RAG response using Watsonx API (copied from original working implementation)."""
        try:
            print("\n🤖 WATSONX RAG (with caching):")
            print(f"1. Query: '{query[:60]}...'")
            
            # Check for cached response first
            if self.cache_manager:
                # Convert SearchResult objects to dict format for cache lookup
                search_results_dict = []
                for result in context:
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
                
                cached_response = self.cache_manager.get_cached_response(
                    query, search_results_dict, "watsonx", language=language
                )
                if cached_response:
                    print("2. ✅ Response cache HIT! Returning cached response")
                    return RAGResponse(
                        response=cached_response,
                        sources=context,
                        metadata={
                            'provider': 'watsonx',
                            'model': self.current_model,
                            'user_id': user_id,
                            'language': language,
                            'cached': True
                        }
                    )
                print("2. ❌ Response cache MISS, generating fresh response")
            else:
                print("2. No cache manager available, generating fresh response")
            
            # Initialize ai_response to avoid variable scope issues
            ai_response = ""
            
            # FIXED: Use AI with actual search results instead of bypassing it completely
            print(f"🔍 DEBUG: Using AI with REAL search results for query: {query}")
            if context:
                print(f"🔍 DEBUG: Processing {len(context)} REAL search results for AI context")
                for i, result in enumerate(context):
                    print(f"🔍 DEBUG: Real Result {i+1}: {result.merchant} - ${result.amount} - {result.description[:50]}...")
            
            insights = self._get_financial_insights(context)
            budget_recommendations = self._generate_budget_recommendations(insights, query)

            if not self.api_key or not self.project_id:
                table_text = ""
                if context:
                    table_text = "\n".join([
                        f"• **{result.metadata.get('shopping_type', 'Unknown')}** at {result.merchant}: ${result.amount} on {result.date} ({result.metadata.get('payment_method', 'Unknown')}) - {result.description}"
                        for result in context
                    ])

                ai_response = f"""## Financial Analysis for: "{query}"

### 📋 Transaction Details
{table_text if table_text else 'No expense records found for this query.'}

### 📊 Financial Summary
• Total Amount: ${insights.get('total_amount', 0):.2f}
• Number of Transactions: {insights.get('num_transactions', 0)}
• Average Transaction: ${insights.get('avg_transaction', 0):.2f}

### 🤖 Recommendations
{budget_recommendations if budget_recommendations else '• Consider reviewing your spending patterns regularly'}"""
            else:
                try:
                    search_results_text = ""
                    if context:
                        search_results_text = "\n".join(
                            f"• **{result.metadata.get('shopping_type', 'Unknown')}** at {result.merchant}: ${result.amount} ({result.metadata.get('payment_method', 'Unknown')}) - {result.description}"
                            for result in context
                        )
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

                    messages = [{"role": "user", "content": enhanced_prompt}]
                    ai_response = self._call_watsonx_api(messages)

                except Exception:
                    ai_response = f"## Financial Analysis for: \"{query}\"\n\n"
                    ai_response += f"Found {len(context)} relevant records.\n"
                    ai_response += f"Total: ${insights.get('total_amount', 0):.2f} across {insights.get('num_transactions', 0)} transactions.\n\n"
                    ai_response += budget_recommendations if budget_recommendations else "• Monitor your spending patterns regularly"
                    ai_response += "\n\n**Note**: API call failed, showing structured analysis."
            
            # Cache the response for future similar queries
            if self.cache_manager and ai_response:
                # Convert SearchResult objects to dict format for caching
                search_results_dict = []
                for result in context:
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
                
                # Estimate token usage (rough approximation)
                prompt_tokens = len(query.split()) * 1.3  # ~1.3 tokens per word
                response_tokens = len(ai_response.split()) * 1.3
                
                if "API call failed" not in ai_response:
                    self.cache_manager.cache_response(
                        query, ai_response, search_results_dict, "watsonx",
                        int(prompt_tokens), int(response_tokens),
                        language=language
                    )
                print(f"3. ✅ Cached response (est. {int(prompt_tokens + response_tokens)} tokens)")
            
            return RAGResponse(
                response=ai_response,
                sources=context,
                metadata={
                    'provider': 'watsonx',
                    'model': self.current_model,
                    'user_id': user_id,
                    'language': language
                }
            )

        except Exception as e:
            from .display import NEUTRAL_UNAVAILABLE
            print(f"Sorry, I'm experiencing technical difficulties. Error: {str(e)}")
            return RAGResponse(
                response=NEUTRAL_UNAVAILABLE,
                sources=context,
                metadata={
                    'provider': 'watsonx',
                    'model': self.current_model,
                    'user_id': user_id,
                    'language': language,
                    'error': str(e)
                }
            )