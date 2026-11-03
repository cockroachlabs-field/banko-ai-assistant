"""
Banko AI Assistant - AI-powered expense analysis and RAG system.

A modern Python package for AI-powered expense analysis using CockroachDB vector search
and multi-provider AI support (OpenAI, AWS Bedrock, IBM Watsonx, Google Gemini).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("banko-ai-assistant")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"

__author__ = "Virag Tripathi"
__email__ = "virag.tripathi@gmail.com"

from .config.settings import Config


def create_app():
    """Create Flask application (lazy import)."""
    from .web.app import create_app as _create_app
    return _create_app()

__all__ = ["Config", "create_app", "__version__"]
