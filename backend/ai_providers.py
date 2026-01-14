"""
AI Provider Abstraction for KEEP Platform
- OpenRouter as primary provider (access to 400+ models)
- Google Gemini Flash as fallback
- Automatic failover with logging
"""
import os
import logging
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

from openai import OpenAI
import google.generativeai as genai

from backend.config import settings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIProviderError(Exception):
    """Custom exception for AI provider failures"""
    pass


class BaseAIClient(ABC):
    """Abstract base class for AI clients"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate content from a prompt"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured and available"""
        pass


class OpenRouterClient(BaseAIClient):
    """
    OpenRouter API client using OpenAI SDK compatibility.
    
    OpenRouter provides access to 400+ models through a unified API.
    Uses OpenAI SDK with custom base_url for seamless integration.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model = model or getattr(settings, 'OPENROUTER_MODEL', 'google/gemini-2.0-flash-001')
        self.base_url = base_url
        self._client = None
        
        if self.api_key:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers={
                    "HTTP-Referer": "https://keep-health.app",  # Required by OpenRouter
                    "X-Title": "KEEP Health Platform"
                }
            )
    
    def is_available(self) -> bool:
        """Check if OpenRouter is configured"""
        return bool(self.api_key and self._client)
    
    def generate(self, prompt: str) -> str:
        """Generate content using OpenRouter API"""
        if not self.is_available():
            raise AIProviderError("OpenRouter client not configured (missing API key)")
        
        try:
            logger.info(f"OpenRouter: Generating content with model {self.model}")
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096
            )
            
            result = response.choices[0].message.content
            logger.info(f"OpenRouter: Successfully generated response ({len(result)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            raise AIProviderError(f"OpenRouter API error: {e}")


class GeminiClient(BaseAIClient):
    """
    Google Gemini API client (fallback provider).
    
    Uses the official google-generativeai SDK.
    Model: gemini-2.0-flash-001 (GA production version as of Feb 2025)
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None):
        self.api_key = api_key or settings.GOOGLE_API_KEY
        self.model_name = model or getattr(settings, 'GEMINI_MODEL', 'gemini-2.0-flash-001')
        self._model = None
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
    
    def is_available(self) -> bool:
        """Check if Gemini is configured"""
        return bool(self.api_key and self._model)
    
    def generate(self, prompt: str) -> str:
        """Generate content using Google Gemini API"""
        if not self.is_available():
            raise AIProviderError("Gemini client not configured (missing API key)")
        
        try:
            logger.info(f"Gemini: Generating content with model {self.model_name}")
            
            response = self._model.generate_content(prompt)
            result = response.text
            
            logger.info(f"Gemini: Successfully generated response ({len(result)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise AIProviderError(f"Gemini API error: {e}")


class AIProvider:
    """
    Unified AI provider with automatic fallback.
    
    Priority order:
    1. OpenRouter (primary) - access to 400+ models
    2. Google Gemini (fallback) - direct API
    
    Automatically falls back to secondary provider if primary fails.
    """
    
    def __init__(self):
        # Initialize providers
        self.openrouter = OpenRouterClient()
        self.gemini = GeminiClient()
        
        # Get priority order from settings
        priority_str = getattr(settings, 'AI_PROVIDER_PRIORITY', 'openrouter,gemini')
        self.priority = [p.strip().lower() for p in priority_str.split(',')]
        
        # Build ordered provider list
        self.providers: List[tuple[str, BaseAIClient]] = []
        for provider_name in self.priority:
            if provider_name == 'openrouter' and self.openrouter.is_available():
                self.providers.append(('OpenRouter', self.openrouter))
            elif provider_name == 'gemini' and self.gemini.is_available():
                self.providers.append(('Gemini', self.gemini))
        
        if not self.providers:
            logger.error("No AI providers available! Check your API keys.")
        else:
            provider_names = [p[0] for p in self.providers]
            logger.info(f"AI Providers initialized: {' -> '.join(provider_names)}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate content with automatic fallback.
        
        Tries providers in priority order. If primary fails,
        automatically falls back to secondary provider.
        
        Args:
            prompt: The input prompt for generation
            
        Returns:
            Generated text content
            
        Raises:
            AIProviderError: If all providers fail
        """
        errors = []
        
        for provider_name, provider in self.providers:
            try:
                result = provider.generate(prompt)
                return result
            except AIProviderError as e:
                error_msg = f"{provider_name}: {e}"
                errors.append(error_msg)
                logger.warning(f"{provider_name} failed, trying next provider...")
                continue
            except Exception as e:
                error_msg = f"{provider_name}: Unexpected error - {e}"
                errors.append(error_msg)
                logger.warning(f"{provider_name} failed unexpectedly, trying next provider...")
                continue
        
        # All providers failed
        error_summary = "; ".join(errors)
        logger.error(f"All AI providers failed: {error_summary}")
        raise AIProviderError(f"All AI providers unavailable. Errors: {error_summary}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all configured providers"""
        return {
            "openrouter": {
                "available": self.openrouter.is_available(),
                "model": self.openrouter.model if self.openrouter.is_available() else None
            },
            "gemini": {
                "available": self.gemini.is_available(),
                "model": self.gemini.model_name if self.gemini.is_available() else None
            },
            "priority": self.priority,
            "active_providers": [p[0] for p in self.providers]
        }


# Singleton instance
_ai_provider: Optional[AIProvider] = None


def get_ai_provider() -> AIProvider:
    """
    Get the singleton AI provider instance.
    
    Creates the provider on first call, reuses on subsequent calls.
    
    Returns:
        AIProvider: The configured AI provider with fallback support
    """
    global _ai_provider
    if _ai_provider is None:
        _ai_provider = AIProvider()
    return _ai_provider


def reset_ai_provider() -> None:
    """Reset the AI provider (useful for testing or config changes)"""
    global _ai_provider
    _ai_provider = None
