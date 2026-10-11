"""
AWS Bedrock AI provider implementation.

This module provides AWS Bedrock integration for vector search and RAG responses.
"""

import json
from typing import Any

import boto3
from sqlalchemy import text

from banko_ai.utils.embeddings import load_embedding_model

from ..config.settings import get_config
from ..utils.db_retry import TRANSIENT_ERRORS, create_resilient_engine, db_retry, get_database_url
from .base import AIConnectionError, AIProvider, RAGResponse, SearchResult
from .rag_prompts import build_banko_rag_prompt


class AWSProvider(AIProvider):
    """AWS Bedrock AI provider implementation."""
    
    def __init__(self, config: dict[str, Any], cache_manager=None):
        """Initialize AWS provider."""
        # Support both config and environment variables with defaults
        cfg = get_config()
        self.access_key_id = config.get("access_key_id") or cfg.aws_access_key_id
        self.secret_access_key = config.get("secret_access_key") or cfg.aws_secret_access_key
        self.region = config.get("region") or cfg.aws_region
        self.profile_name = config.get("profile_name") or cfg.aws_profile
        self.model_id = config.get("model") or cfg.aws_model
        
        self.bedrock_client = None
        self.embedding_model = None
        self.db_engine = None
        self.cache_manager = cache_manager
        self.session = None
        
        # Make credentials optional for demo mode
        if not self.access_key_id and not self.profile_name:
            print("⚠️ AWS_ACCESS_KEY_ID or AWS_PROFILE not found - running in demo mode")
        if not self.secret_access_key and not self.profile_name:
            print("⚠️ AWS_SECRET_ACCESS_KEY or AWS_PROFILE not found - running in demo mode")
        
        super().__init__(config)
    
    def _validate_config(self) -> None:
        """Validate AWS configuration."""
        try:
            # Create a boto3 session with profile if specified, otherwise use default credentials
            if self.profile_name:
                print(f"🔐 Using AWS profile: {self.profile_name}")
                self.session = boto3.Session(
                    profile_name=self.profile_name,
                    region_name=self.region
                )
            elif self.access_key_id and self.secret_access_key:
                print("🔐 Using AWS credentials from environment/config")
                self.session = boto3.Session(
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    region_name=self.region
                )
            else:
                print("🔐 Using default AWS credential chain")
                self.session = boto3.Session(region_name=self.region)
            
            # Create Bedrock client from session
            self.bedrock_client = self.session.client('bedrock-runtime')
            print(f"✅ AWS Bedrock client created (region: {self.region}, model: {self.model_id})")
            
            # Verify credentials by getting caller identity
            print("🔍 Verifying AWS credentials...")
            sts = self.session.client('sts')
            identity = sts.get_caller_identity()
            print(f"✅ AWS Identity verified: {identity['Arn']}")
            print(f"   Account: {identity['Account']}")
            
        except Exception as e:
            error_msg = str(e)
            print("\n⚠️ AWS Bedrock initialization failed!")
            print(f"   Error: {error_msg}")
            
            # Provide specific help based on error type
            if 'ExpiredToken' in error_msg or 'expired' in error_msg.lower():
                print("\n❌ Your AWS credentials have EXPIRED")
                print("\n💡 To fix:")
                print("   1. Get fresh credentials from AWS Console")
                print("   2. Or run: aws sso login --profile CRLRevenue-337380398238")
            elif 'NoCredentials' in error_msg or 'Unable to locate credentials' in error_msg:
                print("\n❌ No AWS credentials found")
                print("\n💡 Set credentials via:")
                print("   - export AWS_PROFILE=CRLRevenue-337380398238")
                print("   - export AWS_ACCESS_KEY_ID=...")
                print("   - export AWS_SECRET_ACCESS_KEY=...")
                print("   - Or configure ~/.aws/credentials")
            elif 'InvalidClientTokenId' in error_msg:
                print("\n❌ Invalid AWS credentials")
                print("\n💡 If using SSO, try:")
                print("   - export AWS_PROFILE=CRLRevenue-337380398238")
                print("   - Then restart the application")
            
            self.bedrock_client = None
            print()
    
    def get_default_model(self) -> str:
        """Get the default AWS model.

        Returns Claude Haiku 4.5 — the current Anthropic Haiku on Bedrock.
        Claude 3.x inference profiles are flagged Legacy after 30 days of
        per-account inactivity (the call returns ResourceNotFoundException
        with a Legacy-upgrade message), so we never seed defaults from that
        generation. See get_available_models() for the dropdown filter that
        also hides Claude 3.x from operators.
        """
        return "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    
    def get_available_models(self) -> list[str]:
        """Discover Claude inference profiles for the current AWS account.

        Filters out Claude 3.x because Bedrock marks those profiles Legacy
        after 30 days of per-account inactivity and InvokeModel returns
        ResourceNotFoundException("upgrade to an active model"). The list API
        still returns them as lifecycle=ACTIVE, so we have to filter by ID
        pattern. Operators who genuinely need a Claude 3.x profile can pin one
        via the AWS_MODELS env var (comma-separated), which short-circuits
        discovery entirely.
        """
        extra = get_config().aws_models
        if extra:
            return [m.strip() for m in extra.split(",") if m.strip()]

        models: set[str] = set()
        try:
            bedrock_client = boto3.client('bedrock', region_name=self.region)

            # Foundation models
            response = bedrock_client.list_foundation_models(byProvider='Anthropic')
            for m in response.get('modelSummaries', []):
                mid = m.get('modelId', '')
                status = m.get('modelLifecycle', {}).get('status', '')
                if status == 'ACTIVE' and 'claude' in mid.lower() and not self._is_legacy_claude(mid):
                    models.add(mid)

            # Inference profiles (required for on-demand invocation of newer models)
            response = bedrock_client.list_inference_profiles()
            for p in response.get('inferenceProfileSummaries', []):
                pid = p.get('inferenceProfileId', '')
                if p.get('status') == 'ACTIVE' and 'claude' in pid.lower() and not self._is_legacy_claude(pid):
                    models.add(pid)

            if models:
                return sorted(models)
        except Exception as e:
            print(f"⚠️  Could not list Bedrock models: {e}")

        # Fallback: known-good current Claude 4.x inference profiles. Refreshed
        # 2026-05-22 against a CRLRevenue Bedrock account; pick Haiku as the
        # demo default to keep token cost low.
        return [
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us.anthropic.claude-sonnet-4-6",
            "us.anthropic.claude-opus-4-7",
        ]

    @staticmethod
    def _is_legacy_claude(model_id: str) -> bool:
        """True for Claude 3.x inference profiles, which Bedrock denies after
        30 days of per-account inactivity. Match on the canonical
        '...claude-3-' substring so both foundation IDs and us./global.
        inference-profile prefixes are covered."""
        return "claude-3-" in model_id.lower()
    
    def _get_embedding_model(self):
        """Get or create the embedding model."""
        if self.embedding_model is None:
            try:
                # Use configurable embedding model from environment or default
                embedding_model_name = get_config().embedding_model
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
            print("\n🔍 AWS BEDROCK VECTOR SEARCH (with caching):")
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
            
            # FIXED: Use named parameters with a dictionary instead of %s with a list
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
                "limit": limit
            }
            
            # Conditionally add user_id filter if provided
            if user_id:
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
                WHERE user_id = :user_id
                ORDER BY embedding <=> :search_embedding
                LIMIT :limit
                """
                params["user_id"] = user_id
            
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
        """Generate comprehensive financial insights from expense data."""
        if not search_results:
            return {}
        
        total_amount = 0
        categories = {}
        merchants = {}
        payment_methods = {}
        
        for result in search_results:
            # Handle both SearchResult objects and dictionaries
            if hasattr(result, 'amount'):
                amount = float(result.amount)
                merchant = result.merchant
                category = result.metadata.get('shopping_type', 'Unknown') if hasattr(result, 'metadata') and result.metadata else 'Unknown'
                payment = result.metadata.get('payment_method', 'Unknown') if hasattr(result, 'metadata') and result.metadata else 'Unknown'
            else:
                amount = float(result.get('expense_amount', 0))
                merchant = result.get('merchant', 'Unknown')
                category = result.get('shopping_type', 'Unknown')
                payment = result.get('payment_method', 'Unknown')
            
            total_amount += amount
            categories[category] = categories.get(category, 0) + amount
            merchants[merchant] = merchants.get(merchant, 0) + amount
            payment_methods[payment] = payment_methods.get(payment, 0) + amount
        
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
        """Generate personalized budget recommendations based on spending patterns."""
        if not insights:
            return ""
        
        recommendations = []
        
        if insights.get('top_category'):
            category, amount = insights['top_category']
            recommendations.append(f"Your highest spending category is **{category}** at **${amount:.2f}**. Consider setting a monthly budget limit for this category.")
        
        avg_amount = insights.get('avg_transaction', 0)
        if avg_amount > 100:
            recommendations.append(f"Your average transaction is **${avg_amount:.2f}**. Consider reviewing larger purchases to identify potential savings.")
        
        if insights.get('top_merchant'):
            merchant, amount = insights['top_merchant']
            recommendations.append(f"You frequently shop at **{merchant}** (${amount:.2f} total). Look for loyalty programs or discounts at this merchant.")
        
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
        """Generate RAG response using AWS Bedrock."""
        try:
            print("\n🤖 AWS BEDROCK RAG (with caching):")
            print(f"1. Query: '{query[:60]}...'")
            
            # Check for cached response first
            if self.cache_manager:
                # Convert context to dict format for cache lookup (handle both objects and dicts)
                search_results_dict = []
                for result in context:
                    if hasattr(result, 'expense_id'):
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
                        search_results_dict.append(result)
                
                cached_response = self.cache_manager.get_cached_response(
                    query, search_results_dict, "aws", language=language
                )
                if cached_response:
                    print("2. ✅ Response cache HIT! Returning cached response")
                    return RAGResponse(
                        response=cached_response,
                        sources=context,
                        metadata={
                            'provider': 'aws',
                            'model': self.get_default_model(),
                            'user_id': user_id,
                            'language': language,
                            'cached': True
                        }
                    )
                print("2. ❌ Response cache MISS, generating fresh response")
            else:
                print("2. No cache manager available, generating fresh response")
            
            # Generate financial insights
            insights = self._get_financial_insights(context)
            budget_recommendations = self._generate_budget_recommendations(insights, query)
            
            # Prepare the search results context with enhanced analysis
            search_results_text = ""
            if context:
                context_parts = []
                for result in context:
                    if hasattr(result, 'amount'):
                        description = result.description
                        merchant = result.merchant
                        amount = result.amount
                        # Format date properly for SearchResult objects
                        date_value = result.date if hasattr(result, 'date') else None
                        if date_value and hasattr(date_value, 'isoformat'):
                            date = date_value.isoformat()
                        elif date_value and str(date_value) != 'None':
                            date = str(date_value)
                        else:
                            date = 'Unknown'
                        shopping_type = result.metadata.get('shopping_type', 'Unknown') if hasattr(result, 'metadata') and result.metadata else 'Unknown'
                        payment_method = result.metadata.get('payment_method', 'Unknown') if hasattr(result, 'metadata') and result.metadata else 'Unknown'
                    else:
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
            
            # Claude on Bedrock rejects both `temperature` and `top_p` in the same
            # request (ValidationException). Keep temperature; drop top_p.
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "top_k": 250,
                "stop_sequences": [],
                "temperature": 1,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": enhanced_prompt
                            }
                        ]
                    }
                ]
            }
            
            # Convert to JSON format
            body = json.dumps(payload)
            
            # Use current model
            model_id = self.current_model
            
            # Invoke model
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            ai_response = response_body['content'][0]['text']
            
            # Cache the response for future similar queries
            if self.cache_manager and ai_response:
                search_results_dict = []
                for result in context:
                    if hasattr(result, 'expense_id'):
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
                        search_results_dict.append(result)
                
                # Estimate token usage
                prompt_tokens = len(enhanced_prompt.split()) * 1.3
                response_tokens = len(ai_response.split()) * 1.3
                
                if "API call failed" not in ai_response:
                    self.cache_manager.cache_response(
                        query, ai_response, search_results_dict, "aws",
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
                    "provider": "aws",
                    "model": model_id,
                    "region": self.region,
                    "language": language,
                    "user_id": user_id,
                    "cached": False
                }
            )
            
        except Exception as e:
            raise AIConnectionError(f"RAG response generation failed: {str(e)}")
    
    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        try:
            embedding_model = self._get_embedding_model()
            embedding = embedding_model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            raise AIConnectionError(f"Embedding generation failed: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test AWS Bedrock connection by verifying credentials (no tokens consumed)."""
        try:
            sts = boto3.client('sts', region_name=self.region)
            sts.get_caller_identity()
            return True
        except Exception:
            return False