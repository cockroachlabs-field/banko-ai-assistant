"""
Command-line interface for Banko AI Assistant.

This module provides CLI commands for running the application and managing data.
"""


import click

from .config.settings import get_config, set_config


@click.group()
@click.pass_context
def cli(ctx):
    """Banko AI Assistant - AI-powered expense analysis and RAG system.
    
    For detailed configuration and environment variables, run:
      banko-ai help
    
    Quick start:
      export AI_SERVICE=watsonx
      export DATABASE_URL=cockroachdb://root@localhost:26257/banko_ai
      banko-ai run
    """
    pass


@cli.command()
@click.option('--host', default=None, help='Host to bind to')
@click.option('--port', default=None, type=int, help='Port to bind to')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--generate-data', type=int, default=5000, help='Generate sample data before starting (default: 5000 records)')
@click.option('--no-data', is_flag=True, help='Skip data generation and start with empty database')
@click.option('--clear-data', is_flag=True, help='Clear existing data before generating new data')
@click.option('--background', is_flag=True, help='Run in background mode (suppress Flask output)')
def run(host, port, debug, generate_data, no_data, clear_data, background):
    """Run the Banko AI Assistant web application."""
    config = get_config()
    host = host or config.host
    port = port or config.port
    click.echo("🏦 === Banko AI Assistant Starting === 🏦")
    click.echo(f"🤖 AI Service: {config.ai_service}")
    click.echo("=" * 44)

    if no_data:
        # The warm-up's sample-data step reads this setting.
        set_config(config.model_copy(update={"banko_skip_autosetup": True}))
    elif clear_data:
        from .vector_search.generator import EnhancedExpenseGenerator
        click.echo("Clearing existing data...")
        EnhancedExpenseGenerator(config.database_url).clear_expenses()

    from .web.app import create_app
    from .web.server import serve

    app = create_app()
    app.debug = bool(debug or config.debug)  # no reloader: serve() runs in this process
    readiness = app.extensions["banko_readiness"]

    def announce_when_warm():
        # The ready line means ready: it waits for every warm-up step.
        import time
        while any(c["state"] in ("pending", "running") for c in readiness.snapshot().values()):
            time.sleep(0.5)
        click.echo("🎉 Banko AI is ready to help with your finances!")
        failed = [name for name, c in readiness.snapshot().items() if c["state"] == "failed"]
        if failed:
            click.echo(f"⚠️  Not ready: {', '.join(failed)} (details at /api/health)")

    def announce(urls):
        import threading
        click.echo(f"🚀 Starting server on http://localhost:{port}")
        click.echo("=" * 44)
        for url in urls:
            click.echo(f" * Running on {url}")
        if background:
            click.echo("💡 Use 'banko-ai status' to check if it's running")
            click.echo("🛑 Use 'pkill -f banko-ai' to stop the background process")
        threading.Thread(target=announce_when_warm, name="banko-ready-line", daemon=True).start()

    if background:
        import logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
    serve(app, host, port, on_ready=announce)


@cli.command()
@click.option('--count', default=1000, help='Number of records to generate')
@click.option('--user-id', help='User ID for generated records')
@click.option('--clear', is_flag=True, help='Clear existing data before generating')
def generate_data(count, user_id, clear):
    """Generate sample expense data."""
    from .vector_search.generator import EnhancedExpenseGenerator

    config = get_config()
    generator = EnhancedExpenseGenerator(config.database_url)
    
    click.echo(f"Generating {count} expense records...")
    
    generated_count = generator.generate_and_save(
        count=count,
        user_id=user_id,
        clear_existing=clear
    )
    
    click.echo(f"Successfully generated {generated_count} expense records")


@cli.command()
def clear_data():
    """Clear all expense data."""
    from .vector_search.generator import EnhancedExpenseGenerator

    config = get_config()
    generator = EnhancedExpenseGenerator(config.database_url)
    
    if generator.clear_expenses():
        click.echo("Successfully cleared all expense data")
    else:
        click.echo("Failed to clear expense data")


@cli.command()
@click.option('--agents', is_flag=True, help='Clear agent tables (agent_state, agent_decisions, agent_memory, agent_tasks)')
@click.option('--cache', is_flag=True, help='Clear cache tables (vector_search_cache, query_cache)')
@click.option('--all', 'clear_all', is_flag=True, help='Clear both agents and cache')
@click.confirmation_option(prompt='Are you sure you want to clear this data?')
def clear_agents(agents, cache, clear_all):
    """Clear agent state and cache to start fresh.
    
    This is useful when:
    - Testing new agent features
    - Demoing to different audiences
    - Resetting agent learning
    
    Examples:
      banko-ai clear-agents --agents        # Clear only agent tables
      banko-ai clear-agents --cache         # Clear only cache
      banko-ai clear-agents --all           # Clear everything
    """
    from sqlalchemy import text

    from .db.engine import get_engine

    config = get_config()
    engine = get_engine(config.database_url)
    
    tables_cleared = []
    
    try:
        with engine.connect() as conn:
            # Clear agent tables
            if agents or clear_all:
                click.echo("🗑️  Clearing agent tables...")
                
                # Clear in order to avoid foreign key constraints
                agent_tables = [
                    'agent_tasks',
                    'agent_memory', 
                    'agent_decisions',
                    'agent_state'
                ]
                
                for table in agent_tables:
                    try:
                        result = conn.execute(text(f"DELETE FROM {table}"))
                        rows_deleted = result.rowcount
                        conn.commit()
                        tables_cleared.append(f"{table} ({rows_deleted} rows)")
                        click.echo(f"  ✅ Cleared {table}: {rows_deleted} rows")
                    except Exception as e:
                        click.echo(f"  ⚠️  Could not clear {table}: {e}")
            
            # Clear cache tables
            if cache or clear_all:
                click.echo("🗑️  Clearing cache tables...")
                
                cache_tables = [
                    'vector_search_cache',
                    'query_cache'
                ]
                
                for table in cache_tables:
                    try:
                        result = conn.execute(text(f"DELETE FROM {table}"))
                        rows_deleted = result.rowcount
                        conn.commit()
                        tables_cleared.append(f"{table} ({rows_deleted} rows)")
                        click.echo(f"  ✅ Cleared {table}: {rows_deleted} rows")
                    except Exception as e:
                        click.echo(f"  ⚠️  Could not clear {table}: {e}")

        if tables_cleared:
            click.echo(f"\n✅ Successfully cleared: {', '.join(tables_cleared)}")
            click.echo("\n💡 Tip: Agent tables will be repopulated when you upload a new receipt")
        else:
            click.echo("\n⚠️  No tables were cleared. Use --agents, --cache, or --all")

    except Exception as e:
        click.echo(f"\n❌ Error clearing data: {e}")


@cli.command()
def status():
    """Show application status."""
    from .vector_search.generator import EnhancedExpenseGenerator

    config = get_config()
    generator = EnhancedExpenseGenerator(config.database_url)
    
    # Show current configuration
    click.echo("🔧 Current Configuration:")
    click.echo(f"   AI Service: {config.ai_service}")
    click.echo(f"   Database: {config.database_url}")
    
    if config.ai_service == 'gemini':
        click.echo(f"   Google Project: {config.google_project_id}")
        click.echo(f"   Google Model: {config.google_model}")
        click.echo(f"   Google Location: {config.google_location}")
        creds_path = config.google_application_credentials
        api_key = config.google_api_key
        click.echo(f"   Service Account: {'✅' if creds_path else '❌'} {creds_path or 'Not set'}")
        click.echo(f"   API Key Fallback: {'✅' if api_key else '❌'} {'Set' if api_key else 'Not set'}")
    
    click.echo()
    
    # Check database connection
    try:
        count = generator.get_expense_count()
        click.echo(f"✅ Database connected - {count} expense records")
    except Exception as e:
        click.echo(f"❌ Database error: {e}")
        return
    
    # Check AI provider
    try:
        from .ai_providers.factory import AIProviderFactory
        ai_config = config.get_ai_config()
        ai_provider = AIProviderFactory.create_provider(
            config.ai_service, 
            ai_config[config.ai_service]
        )
        
        if ai_provider.test_connection():
            click.echo(f"✅ AI provider ({config.ai_service}) connected")
        else:
            click.echo(f"❌ AI provider ({config.ai_service}) disconnected")
    except Exception as e:
        click.echo(f"❌ AI provider error: {e}")


@cli.command()
@click.option('--host', default=None, help='Host to bind to')
@click.option('--port', default=None, type=int, help='Port to bind to')
@click.option('--generate-data', type=int, default=5000, help='Generate sample data before starting (default: 5000 records)')
@click.option('--no-data', is_flag=True, help='Skip data generation and start with empty database')
@click.option('--clear-data', is_flag=True, help='Clear existing data before generating new data')
@click.pass_context
def start(ctx, host, port, generate_data, no_data, clear_data):
    """Start Banko AI in background mode (equivalent to 'run --background')."""
    ctx.invoke(run, host=host, port=port, debug=False, generate_data=generate_data,
               no_data=no_data, clear_data=clear_data, background=True)


@cli.command()
def help():
    """Show detailed help and setup instructions."""
    click.echo("""
🤖 Banko AI Assistant - Setup Guide
====================================

This is a modern AI-powered expense analysis application with RAG capabilities.

PREREQUISITES:
--------------
- CockroachDB v25.2.4+ (recommended: v25.4.0 as Vector index is GA and enabled by default)
- Vector index feature enabled: SET CLUSTER SETTING feature.vector_index.enabled = true;
- Start single node: cockroach start-single-node --insecure --store=./cockroach-data --listen-addr=localhost:26257 --http-addr=localhost:8080 --background

ENVIRONMENT VARIABLES:
---------------------
Required:
  DATABASE_URL              Database connection string
                            - Single node: cockroachdb://root@localhost:26257/banko_ai
                            - With load balancer: cockroachdb://root@haproxy:26257/banko_ai
  AI_SERVICE                AI provider: watsonx, openai, aws, gemini

IBM Watsonx:
  WATSONX_API_KEY          IBM Cloud API key
  WATSONX_PROJECT_ID       Watsonx project ID
  WATSONX_MODEL_ID         Model to use (default: openai/gpt-oss-120b)
  WATSONX_API_URL          API endpoint (default: us-south region)
  WATSONX_TOKEN_URL        IAM token endpoint
  WATSONX_TIMEOUT          Request timeout in seconds (default: 30)

OpenAI:
  OPENAI_API_KEY           OpenAI API key
  OPENAI_MODEL             Model to use (default: gpt-4o-mini)

AWS Bedrock:
  AWS_ACCESS_KEY_ID        AWS access key
  AWS_SECRET_ACCESS_KEY    AWS secret key
  AWS_REGION               AWS region (default: us-east-1)
  AWS_MODEL_ID             Model to use (default: claude-3-5-sonnet)

Google Gemini:
  GOOGLE_APPLICATION_CREDENTIALS  Path to service account JSON
  GOOGLE_PROJECT_ID        Google Cloud project ID
  GOOGLE_MODEL             Model to use (default: gemini-2.0-flash-001)
  GOOGLE_LOCATION          Region (default: us-central1)
  GOOGLE_API_KEY           API key (fallback if Vertex AI unavailable)

Optional - Global:
  EMBEDDING_MODEL          Embedding model for all providers (default: all-MiniLM-L6-v2)
  FLASK_ENV                Flask environment: development, production
  SECRET_KEY               Flask secret key for sessions

Database Connection Pool (all optional):
  DB_POOL_SIZE             Base connection pool size (default: 100)
  DB_MAX_OVERFLOW          Max overflow connections (default: 100)
  DB_POOL_TIMEOUT          Timeout for connection in seconds (default: 30)
  DB_POOL_RECYCLE          Recycle connections after N seconds (default: 3600)
  DB_POOL_PRE_PING         Test connections before use (default: true)
  DB_CONNECT_TIMEOUT       Database connection timeout (default: 10)
  
  Pool size recommendations:
  - Low traffic (<10 QPS): 10-50 connections
  - Medium (10-100 QPS): 100-500 connections
  - High (100+ QPS): 500-1000+ connections
  - For 1000 connections: DB_POOL_SIZE=500 DB_MAX_OVERFLOW=500

Response Caching (all optional):
  CACHE_SIMILARITY_THRESHOLD  Similarity threshold for cache matching (default: 0.75)
                              Range: 0.0-1.0. Lower = more cache hits, higher = more accuracy
  CACHE_TTL_HOURS             Cache time-to-live in hours (default: 24)
  CACHE_STRICT_MODE           Require exact expense data match (default: true)
                              true = accurate (requires matching data)
                              false = aggressive caching (similarity only)
  
  Caching presets:
  - Demo mode (high cache hits): CACHE_THRESHOLD=0.75 CACHE_STRICT_MODE=false
  - Balanced (recommended): CACHE_THRESHOLD=0.75 CACHE_STRICT_MODE=true
  - Conservative (accuracy): CACHE_THRESHOLD=0.85 CACHE_STRICT_MODE=true

LangGraph Checkpointer:
  CHECKPOINT_TTL_DAYS         Auto-expire checkpoints after N days (default: 7, 0 = disabled)
                              Uses CockroachDB row-level TTL for automatic background cleanup.

QUICK START:
-----------
1. Set up required variables:
   export AI_SERVICE="watsonx"
   export DATABASE_URL="cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"

2. Configure your AI provider:

   For Watsonx (IBM):
   export WATSONX_API_KEY="your_api_key_here"
   export WATSONX_PROJECT_ID="your_project_id_here"
   export WATSONX_MODEL_ID="meta-llama/llama-2-70b-chat"
   
   # Optional - Regional/Advanced:
   export WATSONX_API_URL="https://eu-de.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
   export WATSONX_TIMEOUT="60"

   For OpenAI:
   export OPENAI_API_KEY="your_api_key_here"
   export OPENAI_MODEL="gpt-4o-mini"

   For AWS Bedrock:
   export AWS_ACCESS_KEY_ID="your_access_key"
   export AWS_SECRET_ACCESS_KEY="your_secret_key"
   export AWS_REGION="us-east-1"
   export AWS_MODEL_ID="us.anthropic.claude-3-5-sonnet-20241022-v2:0"

   For Google Gemini:
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
   export GOOGLE_PROJECT_ID="your-google-cloud-project-id"
   export GOOGLE_MODEL="gemini-1.5-pro"
   export GOOGLE_LOCATION="us-central1"
   # Optional fallback:
   export GOOGLE_API_KEY="your-gemini-api-key"

   Optional - Global Configuration:
   export EMBEDDING_MODEL="all-MiniLM-L6-v2"  # Use custom embedding model

3. Start the application:
   banko-ai run                    # Normal mode with full output
   banko-ai start                  # Background mode (quiet) - same as run --background
   banko-ai run --background       # Background mode (quiet)
   banko-ai run --no-data          # Skip data generation
   banko-ai run --debug            # Enable debug mode

COMMANDS:
---------
  run              Run the web application (normal mode)
  start            Start the web application (background mode)
  generate-data    Generate sample expense data
  clear-data       Clear all expense data
  status           Show application status
  search           Search expenses using vector similarity
  help             Show this help message

FEATURES:
---------
✅ Multi-AI Provider Support (OpenAI, AWS Bedrock, IBM Watsonx, Google Gemini)
✅ Dynamic Model Switching (switch models without restart)
✅ User Authentication & User-Specific Vector Indexing
✅ Enhanced Vector Search with Data Enrichment
✅ Modern PyPI-Ready Package Structure
✅ Real-time Chat Interface
✅ Expense Analysis & Categorization
✅ Multi-language Support

ENDPOINTS:
----------
Web Interface: http://localhost:5000
API Health: http://localhost:5000/api/health
AI Providers: http://localhost:5000/api/ai-providers
Models: http://localhost:5000/api/models
Search: POST http://localhost:5000/api/search (JSON: {"query": "your search", "limit": 10})
RAG: POST http://localhost:5000/api/rag (JSON: {"query": "your question", "limit": 5})

TROUBLESHOOTING:
---------------
- Database connection issues: Check DATABASE_URL
- AI provider errors: Verify API keys and configuration
- Model switching: Use the Settings tab in the web interface
- Vector search: Ensure database has expense data

Gemini-Specific Issues:
- "404 Publisher Model not found": Enable Vertex AI API in Google Cloud Console
- "Permission denied to enable service": Use your main Google account, not service account
- Vertex AI unavailable: Provider will auto-fallback to Generative AI API if GOOGLE_API_KEY is set
- Service account issues: Ensure the JSON file path is correct and accessible

For more information, visit: https://github.com/cockroachlabs-field/banko-ai-assistant
""")


@cli.command()
@click.argument('query')
@click.option('--user-id', help='User ID to filter results')
@click.option('--limit', default=10, help='Maximum number of results')
def search(query, user_id, limit):
    """Search expenses using vector similarity."""
    config = get_config()
    from .vector_search.search import VectorSearchEngine

    search_engine = VectorSearchEngine(config.database_url)
    results = search_engine.search_expenses(
        query=query,
        user_id=user_id,
        limit=limit
    )

    if not results:
        click.echo("No results found")
        return

    click.echo(f"Found {len(results)} results for '{query}':")
    click.echo()

    for i, result in enumerate(results, 1):
        click.echo(f"{i}. {result.description}")
        click.echo(f"   Merchant: {result.merchant}")
        click.echo(f"   Amount: ${result.amount:.2f}")
        click.echo(f"   Date: {result.date}")
        click.echo(f"   Similarity: {result.similarity_score:.3f}")
        click.echo()


@cli.command("config")
@click.option("--docs", is_flag=True, help="Print the README configuration reference instead.")
def config_cmd(docs):
    """Show the effective configuration and where each value came from."""
    from .config.settings import GROUPS, render_config_docs

    if docs:
        click.echo(render_config_docs(), nl=False)
        return
    current = None
    for info in get_config().describe():
        if info.group != current:
            current = info.group
            click.echo(f"\n[{GROUPS[info.group]}]")
        click.echo(f"  {info.env:<32} {info.value:<56} ({info.source})")


@cli.command()
@click.option('--yes', is_flag=True, help='Skip confirmation prompt')
def clear_demo_users(yes):
    """Clear all demo users and their data (preserves legacy personas).

    This removes demo users created via signup along with their expenses,
    signals, nudges, and budgets. The three legacy persona users (maya,
    sam, riley) are preserved.
    """
    if not yes:
        if not click.confirm('Delete all demo users and their data?'):
            click.echo('Aborted.')
            return

    config = get_config()
    from .web.auth import clear_demo_users as clear_fn

    click.echo('Clearing demo users...')
    counts = clear_fn(config.database_url)

    click.echo('Deleted:')
    click.echo(f'  Users: {counts["users"]}')
    click.echo(f'  Expenses: {counts["expenses"]}')
    click.echo(f'  User budgets: {counts["user_budgets"]}')
    click.echo(f'  Spending signals: {counts["spending_signals"]}')
    click.echo(f'  Coach nudges: {counts["coach_nudges"]}')


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()
