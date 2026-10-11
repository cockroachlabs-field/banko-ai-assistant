"""
Main entry point for Banko AI Assistant.
This matches the original app.py behavior.
"""

if __name__ == '__main__':
    from .config.settings import get_config
    from .web.app import create_app

    print("🏦 === Banko AI Assistant Starting === 🏦")

    # Create the Flask app
    app = create_app()

    # Get port from environment variable or default to 5000 (matching original)
    port = get_config().port
    
    print(f"🚀 Starting server on http://localhost:{port}")
    print("🎉 Banko AI is ready to help with your finances!")
    print("=" * 45)
    
    # Run the app on all interfaces, using the configured port (matching original)
    app.run(host='0.0.0.0', port=port, debug=True)
