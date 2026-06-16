"""
Search tools for agents.

Provides abstract search interface and implementations for different backends.
"""

from .base import BaseSearchTool, SearchResult, Article
from .chunking import chunk_text, chunk_article
from .handler import SearchHandler
from .openreward import OpenRewardSearchConfig, OpenRewardSearchTool

__all__ = [
    'BaseSearchTool',
    'SearchResult', 
    'Article',
    'chunk_text',
    'chunk_article',
    'SearchHandler',
    'OpenRewardSearchConfig',
    'OpenRewardSearchTool',
]
