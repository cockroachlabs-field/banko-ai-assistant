"""
Enhanced expense data generator with data enrichment.

This module generates realistic expense data with enriched descriptions
for improved vector search accuracy.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text

from ..utils.db_retry import get_database_url
from .enrichment import DataEnricher


class EnhancedExpenseGenerator:
    """Enhanced expense generator with data enrichment for better vector search."""
    
    def __init__(self, database_url: str | None = None):
        """Initialize the enhanced expense generator."""
        self.database_url = get_database_url(database_url)
        self.enricher = DataEnricher()
        self._embedding_model = None
        self._merchants = None
        self._categories = None
        self._payment_methods = None
        self._user_ids = None

    @property
    def engine(self):
        """Get SQLAlchemy engine with resilience settings (lazy import)."""
        from ..db.engine import get_engine
        return get_engine(self.database_url)
    
    @property
    def embedding_model(self):
        """Get embedding model (lazy import)."""
        if self._embedding_model is None:
            from banko_ai.utils.embeddings import load_embedding_model

            from ..config.settings import get_config

            # Use configurable embedding model from environment or default
            embedding_model_name = get_config().embedding_model
            self._embedding_model = load_embedding_model(embedding_model_name)
        return self._embedding_model
    
    @property
    def merchants(self):
        """Get merchants data (lazy load)."""
        if self._merchants is None:
            self._init_merchants_and_categories()
        return self._merchants
    
    @property
    def categories(self):
        """Get categories data (lazy load)."""
        if self._categories is None:
            self._init_merchants_and_categories()
        return self._categories
    
    @property
    def payment_methods(self):
        """Get payment methods (lazy load)."""
        if self._payment_methods is None:
            self._init_merchants_and_categories()
        return self._payment_methods
    
    @property
    def user_ids(self):
        """Get user IDs (lazy load)."""
        if self._user_ids is None:
            self._init_merchants_and_categories()
        return self._user_ids
    
    def _init_merchants_and_categories(self):
        """Initialize merchants and categories data - matches original CSV exactly."""
        # Use the exact merchants from the original CSV
        self._merchants = [
            "Starbucks", "Local Market", "McDonald's", "IKEA", "Amazon", "Whole Foods",
            "Italian Bistro", "Uber", "Lyft", "Spotify", "Delta Airlines", "Costco",
            "Home Depot", "Shell Gas Station", "Lowe's", "Tesla Supercharger", "Planet Fitness",
            "Apple Store", "Walmart", "Target", "Netflix", "Best Buy", "CVS Pharmacy",
            "Walgreens", "Rite Aid", "Chipotle", "Subway", "Pizza Hut", "Domino's",
            "Exxon", "Chevron", "BP", "Dunkin' Donuts", "Peet's Coffee", "Ace Hardware",
            "Movie Theater", "Concert Venue", "Gaming Store", "Electric Company",
            "Internet Provider", "Phone Company", "Water Company"
        ]
        
        # Use the exact categories from the original CSV with appropriate merchants
        self._categories = {
            "Groceries": {
                "items": ["Fresh produce", "Dairy products", "Meat and poultry", "Pantry staples", "Organic foods", "Beverages", "Snacks"],
                "merchants": ["Whole Foods", "Local Market", "Costco", "Walmart", "Target"],
                "amount_range": (10, 150)
            },
            "Home Improvement": {
                "items": ["Tools", "Hardware", "Paint", "Lumber", "Electrical supplies", "Plumbing supplies", "Garden supplies"],
                "merchants": ["Home Depot", "Lowe's", "Ace Hardware", "IKEA"],
                "amount_range": (20, 500)
            },
            "Electronics": {
                "items": ["Smartphone", "Laptop", "Tablet", "Headphones", "Camera", "Gaming console", "Smart home device"],
                "merchants": ["Apple Store", "Best Buy", "Amazon", "Target", "Walmart"],
                "amount_range": (50, 1000)
            },
            "Subscription": {
                "items": ["Streaming service", "Software subscription", "Gym membership", "News subscription", "Cloud storage", "Music service"],
                "merchants": ["Netflix", "Spotify", "Planet Fitness", "Electric Company", "Internet Provider"],
                "amount_range": (10, 50)
            },
            "Shopping": {
                "items": ["Clothing", "Shoes", "Accessories", "Home decor", "Books", "Toys", "Beauty products"],
                "merchants": ["Amazon", "Target", "Walmart", "IKEA", "Best Buy"],
                "amount_range": (15, 200)
            },
            "Restaurant": {
                "items": ["Dinner", "Lunch", "Breakfast", "Takeout", "Delivery", "Catering", "Fine dining"],
                "merchants": ["McDonald's", "Italian Bistro", "Chipotle", "Subway", "Pizza Hut", "Domino's"],
                "amount_range": (15, 100)
            },
            "Transport": {
                "items": ["Uber ride", "Lyft ride", "Taxi", "Bus fare", "Train ticket", "Flight", "Car rental"],
                "merchants": ["Uber", "Lyft", "Delta Airlines"],
                "amount_range": (5, 500)
            },
            "Fuel": {
                "items": ["Gas fill-up", "Electric charging", "Diesel fuel", "Premium gas", "Regular gas"],
                "merchants": ["Shell Gas Station", "Tesla Supercharger", "Exxon", "Chevron", "BP"],
                "amount_range": (20, 100)
            },
            "Travel": {
                "items": ["Flight", "Hotel", "Car rental", "Travel insurance", "Airport parking", "Baggage fee"],
                "merchants": ["Delta Airlines", "Hilton Hotels"],
                "amount_range": (100, 2000)
            },
            "Coffee": {
                "items": ["Coffee", "Espresso", "Latte", "Cappuccino", "Pastry", "Sandwich", "Breakfast"],
                "merchants": ["Starbucks", "Local Market", "Whole Foods", "Costco"],
                "amount_range": (3, 25)
            }
        }
        
        # Use the exact payment methods from the original CSV
        self._payment_methods = [
            "Debit Card", "PayPal", "Apple Pay", "Bank Transfer", "Credit Card"
        ]
        self._user_ids = [str(uuid.uuid4()) for _ in range(100)]  # Generate 100 user IDs
    
    def generate_expense(self, user_id: str | None = None) -> dict[str, Any]:
        """Generate a single enriched expense record that matches the original CSV format."""
        # Select category and get associated data
        category = random.choice(list(self.categories.keys()))
        category_data = self.categories[category]
        
        # Select merchant from category-specific merchants
        merchant = random.choice(category_data["merchants"])
        
        # Generate amount within category range
        amount = round(random.uniform(*category_data["amount_range"]), 2)
        
        # Select item from category items
        item = random.choice(category_data["items"])
        
        # Generate basic description
        f"Bought {item.lower()}"
        
        # Generate date (last 90 days)
        days_ago = random.randint(0, 90)
        expense_date = (datetime.now() - timedelta(days=days_ago)).date()
        
        # Generate additional metadata
        payment_method = random.choice(self.payment_methods)
        recurring = random.choice([True, False]) if category in ["Subscription", "Coffee"] else False
        tags = [category.lower(), merchant.lower().replace(" ", "_")]
        
        # Create the exact same description format as the original CSV
        enriched_description = f"Spent ${amount:.2f} on {category.lower()} at {merchant} using {payment_method}."
        
        # Create searchable text for embedding (same as description for simplicity)
        searchable_text = enriched_description
        
        # Generate embedding
        embedding = self.embedding_model.encode([searchable_text])[0].tolist()
        
        return {
            "expense_id": str(uuid.uuid4()),
            "user_id": user_id or random.choice(self.user_ids),
            "expense_date": expense_date,
            "expense_amount": amount,
            "shopping_type": category,
            "description": enriched_description,
            "merchant": merchant,
            "payment_method": payment_method,
            "recurring": recurring,
            "tags": tags,
            "embedding": embedding,
            "searchable_text": searchable_text  # Store for debugging
        }
    
    def generate_expenses(self, count: int, user_id: str | None = None) -> list[dict[str, Any]]:
        """Generate multiple enriched expense records with batch embedding for performance."""
        import time
        print(f"🚀 Generating {count} expense records with batch embedding...")
        start_time = time.time()
        
        # Step 1: Generate expense data WITHOUT embeddings (fast)
        expenses = []
        print("📝 Step 1/2: Generating expense data...")
        for i in range(count):
            expense = self._generate_expense_without_embedding(user_id)
            expenses.append(expense)
            
            # Show progress every 10%
            if (i + 1) % max(1, count // 10) == 0:
                progress = ((i + 1) / count) * 100
                print(f"   Progress: {progress:.0f}% ({i + 1}/{count} records)")
        
        data_gen_time = time.time() - start_time
        print(f"✅ Data generation completed in {data_gen_time:.2f}s")
        
        # Step 2: Generate embeddings in batches (much faster)
        print("🧠 Step 2/2: Generating embeddings in batches...")
        embedding_start = time.time()
        batch_size = 1000  # Process 1000 embeddings at once
        
        for i in range(0, len(expenses), batch_size):
            batch = expenses[i:i + batch_size]
            texts = [exp['searchable_text'] for exp in batch]
            
            # Generate embeddings for entire batch at once
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            # Assign embeddings back to expenses
            for j, embedding in enumerate(embeddings):
                expenses[i + j]['embedding'] = embedding.tolist()
            
            # Show progress
            progress = min(100, ((i + batch_size) / len(expenses)) * 100)
            print(f"   Progress: {progress:.0f}% ({min(i + batch_size, len(expenses))}/{len(expenses)} embeddings)")
        
        embedding_time = time.time() - embedding_start
        total_time = time.time() - start_time
        
        print(f"✅ Embedding generation completed in {embedding_time:.2f}s")
        print(f"🎉 Total generation time: {total_time:.2f}s ({count/total_time:.0f} records/sec)")
        
        return expenses
    
    def _generate_expense_without_embedding(self, user_id: str | None = None) -> dict[str, Any]:
        """Generate a single expense record WITHOUT embedding (for batch processing)."""
        # Select category and get associated data
        category = random.choice(list(self.categories.keys()))
        category_data = self.categories[category]
        
        # Select merchant from category-specific merchants
        merchant = random.choice(category_data["merchants"])
        
        # Generate amount within category range
        amount = round(random.uniform(*category_data["amount_range"]), 2)
        
        # Select item from category items
        random.choice(category_data["items"])
        
        # Generate date (last 90 days)
        days_ago = random.randint(0, 90)
        expense_date = (datetime.now() - timedelta(days=days_ago)).date()
        
        # Generate additional metadata
        payment_method = random.choice(self.payment_methods)
        recurring = random.choice([True, False]) if category in ["Subscription", "Coffee"] else False
        tags = [category.lower(), merchant.lower().replace(" ", "_")]
        
        # Create enriched description
        enriched_description = f"Spent ${amount:.2f} on {category.lower()} at {merchant} using {payment_method}."
        searchable_text = enriched_description
        
        return {
            "expense_id": str(uuid.uuid4()),
            "user_id": user_id or random.choice(self.user_ids),
            "expense_date": expense_date,
            "expense_amount": amount,
            "shopping_type": category,
            "description": enriched_description,
            "merchant": merchant,
            "payment_method": payment_method,
            "recurring": recurring,
            "tags": tags,
            "embedding": None,  # Will be filled in batch
            "searchable_text": searchable_text
        }
    
    def save_expenses_to_database(self, expenses: list[dict[str, Any]]) -> int:
        """Save expenses to the database with retry logic for CockroachDB and multi-region failover."""
        import random
        import time

        import pandas as pd
        from sqlalchemy.exc import DBAPIError, OperationalError

        from ..utils.db_retry import is_transient_error
        from ..utils.migration import regional_tables_ready
        from ..web.auth import resolve_user_region

        is_regional = regional_tables_ready(self.database_url)

        # Prepare data for insertion
        data_to_insert = []
        for expense in expenses:
            row = {
                'expense_id': expense['expense_id'],
                'user_id': expense['user_id'],
                'expense_date': expense['expense_date'],
                'expense_amount': expense['expense_amount'],
                'shopping_type': expense['shopping_type'],
                'description': expense['description'],
                'merchant': expense['merchant'],
                'payment_method': expense['payment_method'],
                'recurring': expense['recurring'],
                'tags': expense['tags'],
                'embedding': expense['embedding']
            }
            if is_regional:
                user_region = resolve_user_region(expense['user_id'], self.database_url)
                if user_region:
                    row['crdb_region'] = user_region
            data_to_insert.append(row)
        
        # Insert in smaller batches to reduce transaction conflicts
        # Use environment variable or default to 50
        from ..config.settings import get_config

        batch_size = get_config().data_gen_batch_size
        total_inserted = 0
        total_batches = (len(data_to_insert) + batch_size - 1) // batch_size
        
        print(f"📊 Inserting {len(data_to_insert)} records in {total_batches} batches of {batch_size}")
        
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i + batch_size]

            # Retry logic for CockroachDB transaction conflicts and multi-region failover
            max_retries = 10  # Increased for multi-region scenarios
            retry_count = 0

            while retry_count < max_retries:
                try:
                    with self.engine.begin() as conn:
                        # Use pandas to insert the batch
                        df = pd.DataFrame(batch)
                        # Specify dtypes for all columns to avoid psycopg3 type casting issues
                        from sqlalchemy import ARRAY, Boolean, Date, Numeric, Text
                        from sqlalchemy.dialects.postgresql import UUID
                        dtype = {
                            'expense_id': UUID(as_uuid=True),
                            'user_id': UUID(as_uuid=True),
                            'expense_date': Date(),
                            'expense_amount': Numeric(precision=10, scale=2),
                            'shopping_type': Text(),
                            'description': Text(),
                            'merchant': Text(),
                            'payment_method': Text(),
                            'recurring': Boolean(),
                            'tags': ARRAY(Text()),
                            'embedding': ARRAY(Numeric(precision=18, scale=15)),
                        }
                        df.to_sql('expenses', conn, if_exists='append', index=False, method='multi', dtype=dtype)
                        # Transaction is automatically committed when exiting the context
                        
                    # Only increment counter after successful transaction
                    total_inserted += len(batch)
                    batch_num = i//batch_size + 1
                    print(f"✅ Batch {batch_num}/{total_batches}: {len(batch)} records inserted (Total: {total_inserted})")
                    break  # Success, exit retry loop
                        
                except (OperationalError, DBAPIError) as e:
                    # Check if it's a transient error (connection issues, serialization conflicts, region failures)
                    if is_transient_error(e):
                        retry_count += 1
                        if retry_count < max_retries:
                            # Exponential backoff with jitter
                            base_delay = 0.1 * (2 ** retry_count)
                            jitter = random.uniform(0, 0.1)
                            delay = base_delay + jitter
                            
                            # Determine error type for better user feedback
                            error_str = str(e).lower()
                            if "connection" in error_str or "failed to connect" in error_str:
                                error_type = "Connection/Region"
                            elif "ambiguous" in error_str or "statementcompletionunknown" in error_str:
                                error_type = "Multi-region failover"
                            else:
                                error_type = "Transaction"
                            
                            print(f"⚠️  {error_type} error detected, retrying in {delay:.2f}s (attempt {retry_count}/{max_retries})")
                            print(f"   Error: {str(e)[:150]}...")
                            time.sleep(delay)
                            continue
                        else:
                            print(f"❌ Max retries ({max_retries}) exceeded for batch {i//batch_size + 1}")
                            print(f"   Last error: {e}")
                            return total_inserted
                    else:
                        # Non-retryable error
                        print(f"❌ Non-retryable database error: {e}")
                        return total_inserted
                        
                except Exception as e:
                    print(f"Unexpected error saving batch {i//batch_size + 1}: {e}")
                    return total_inserted
        
        return total_inserted
    
    def clear_expenses(self) -> bool:
        """Clear the sample expenses, with retry logic. Persona expenses stay:
        their ledger entries point at them, and deleting one half of a
        persona's history would leave the other half dangling."""
        import random
        import time

        from sqlalchemy import text
        from sqlalchemy.exc import DBAPIError, OperationalError

        from ..config.catalog import CatalogError, get_catalog
        from ..utils.db_retry import is_transient_error

        try:
            personas = [p.user_id for p in get_catalog().personas]
        except CatalogError:
            personas = []
        max_retries = 10  # Increased for multi-region scenarios
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self.engine.begin() as conn:
                    conn.execute(text("DELETE FROM expenses WHERE user_id IS NULL "
                                      "OR user_id <> ALL(CAST(:personas AS UUID[]))"),
                                 {"personas": personas})
                    # Transaction is automatically committed when exiting the context
                    return True
                    
            except (OperationalError, DBAPIError) as e:
                # Check if it's a transient error (connection issues, serialization conflicts, region failures)
                if is_transient_error(e):
                    retry_count += 1
                    if retry_count < max_retries:
                        # Exponential backoff with jitter
                        base_delay = 0.1 * (2 ** retry_count)
                        jitter = random.uniform(0, 0.1)
                        delay = base_delay + jitter
                        print(f"⚠️  Transient error while clearing, retrying in {delay:.2f}s (attempt {retry_count}/{max_retries})")
                        print(f"   Error: {str(e)[:150]}...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"❌ Max retries exceeded while clearing expenses: {e}")
                        return False
                else:
                    # Non-retryable error
                    print(f"❌ Non-retryable database error while clearing: {e}")
                    return False
                    
            except Exception as e:
                print(f"Unexpected error clearing expenses: {e}")
                return False
        
        return False
    
    def get_expense_count(self) -> int:
        """Get the current number of expenses in the database."""
        try:
            # Ensure tables exist first
            self._ensure_tables_exist()
            
            from sqlalchemy import text
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM expenses"))
                return result.scalar()
        except Exception as e:
            print(f"Error getting expense count: {e}")
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                print("💡 Make sure CockroachDB is running:")
                print("   cockroach start --insecure")
                print("   Or set DATABASE_URL to your database connection string")
            return 0
    
    def _ensure_tables_exist(self):
        """Ensure database tables exist with correct schema."""
        from sqlalchemy import text

        from ..utils.database import DatabaseManager
        
        print("   Checking expenses table schema...")
        db_manager = DatabaseManager(self.database_url)
        
        # Check if expenses table exists and has correct schema
        if db_manager.table_exists('expenses'):
            print("   ✓ Table exists, verifying schema...")
            # Check embedding column type
            with db_manager.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT column_name, data_type, udt_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'expenses' AND column_name = 'embedding'
                """))
                row = result.fetchone()
                if row:
                    data_type = row[2] if row[2] else row[1]  # Use udt_name first
                    print(f"   ✓ embedding column type: {data_type}")
                    if 'vector' not in data_type.lower():
                        print("\n❌ ERROR: expenses table has WRONG schema!")
                        print(f"   embedding column is '{data_type}', should be 'VECTOR(384)'")
                        print("   This will cause vector search to fail!")
                        print("\n   To fix, run: DROP TABLE expenses CASCADE;")
                        print("   Then re-run this command.\n")
                        raise Exception(f"Invalid schema: embedding is {data_type}, not VECTOR(384)")
                else:
                    print("   ⚠️  embedding column not found!")
        else:
            print("   ✓ Table doesn't exist, will create with correct schema...")
        
        # Create tables with correct schema (IF NOT EXISTS will skip if already exists)
        print("   Creating/verifying table structure...")
        success = db_manager.create_tables()
        if success:
            print("   ✅ Table structure verified/created successfully")
        else:
            print("   ⚠️  Table creation returned False (may already exist with correct schema)")
    
    def generate_and_save(
        self, 
        count: int, 
        user_id: str | None = None, 
        clear_existing: bool = False
    ) -> int:
        """Generate and save expenses to the database."""
        print("\n" + "="*60)
        print("🔧 ENSURING CORRECT DATABASE SCHEMA BEFORE DATA GENERATION")
        print("="*60)
        
        # CRITICAL: Ensure table with correct schema exists BEFORE generating data
        self._ensure_tables_exist()
        
        print("✅ Schema verification complete")
        print("="*60 + "\n")
        
        if clear_existing:
            print("🗑️  Clearing existing expenses...")
            self.clear_expenses()
        
        expenses = self.generate_expenses(count, user_id)
        saved = self.save_expenses_to_database(expenses)
        return saved

    def create_user_specific_indexes(self) -> bool:
        """Create user-specific vector indexes for CockroachDB."""
        try:
            with self.engine.connect() as conn:
                # Create user-specific vector index
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_expenses_user_embedding 
                    ON expenses (user_id, embedding) 
                    USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = 100)
                """))
                
                # Create regional index if supported
                try:
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_expenses_user_embedding_regional 
                        ON expenses (user_id, embedding) 
                        LOCALITY REGIONAL BY ROW AS region
                    """))
                except Exception:
                    # Regional indexing might not be supported in all deployments
                    pass
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error creating user-specific indexes: {e}")
            return False
