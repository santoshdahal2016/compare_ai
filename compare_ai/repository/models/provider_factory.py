from pathlib import Path
import importlib
import functools
from typing import Dict, Set, Any, Optional

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Union
from .models import Model, ModelCapability
from .definitions import TaskType
from .provider import Provider


class ProviderFactory:
    """Factory class for dynamically loading and creating AI model providers.
    
    This factory handles the dynamic loading of provider implementations based on a 
    consistent naming convention. It supports lazy loading of providers and their
    dependencies, making it easy to add new providers without modifying existing code.
    
    Naming Convention:
        - Module files should be named: {provider_name}_provider.py
        - Classes should be named: {ProviderName}Provider
        
    Example:
        ```python
        # List supported providers
        providers = ProviderFactory.get_supported_providers()
        print(f"Available providers: {providers}")
        
        # Create a provider with configuration
        config = {"api_key": "your-api-key"}
        provider = ProviderFactory.create_provider("openai", config)
        

        ```
    """

    PROVIDERS_DIR = Path(__file__).parent / "providers"

    @classmethod
    def create_provider(cls, provider_key: str, config: Optional[Dict[str, Any]] = None) -> Provider:
        """Dynamically load and create an instance of a provider.
        
        This method handles the dynamic import and instantiation of provider classes
        based on the provider_key. It follows a consistent naming convention and
        handles dependency management.
        
        Args:
            provider_key: String identifier for the provider (e.g., 'openai', 'anthropic')
            config: Optional configuration dictionary for the provider initialization
                   Example: {"api_key": "abc123", "organization": "org-123"}
            
        Returns:
            Provider: An instantiated provider object
            
        Raises:
            ImportError: If the provider's required packages are not installed
            ValueError: If the provider is not supported or configuration is invalid
            
        Example:
            ```python
            config = {
                "api_key": "your-api-key",
                "organization": "your-org"
            }
            provider = ProviderFactory.create_provider("openai", config)
            ```
        """
        if provider_key not in cls.get_supported_providers():
            raise ValueError(
                f"Provider '{provider_key}' is not supported. "
                f"Supported providers: {cls.get_supported_providers()}"
            )

        provider_class_name = f"{provider_key.capitalize()}Provider"
        provider_module_name = f"{provider_key}_provider"
        module_path = f"compare_ai.repository.models.providers.{provider_module_name}"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not import provider '{provider_key}'. "
                f"Please install the required package with: poetry install --extras {provider_key}"
            ) from e

        provider_class = getattr(module, provider_class_name)
        config = config or {}
        return provider_class(**config)

    @classmethod
    @functools.cache
    def get_supported_providers(cls) -> Set[str]:
        """Get the set of supported provider names.
        
        This method scans the providers directory to find all available provider
        implementations. The results are cached for performance.
        
        Returns:
            Set[str]: Set of provider names (e.g., {'openai', 'anthropic'})
            
        Example:
            ```python
            providers = ProviderFactory.get_supported_providers()
            print(f"Available providers: {providers}")
            ```
        """
        provider_files = cls.PROVIDERS_DIR.glob("*_provider.py")
        return {
            file.stem.replace("_provider", "") 
            for file in provider_files 
            if not file.stem.startswith("_")
        }

    @classmethod
    def is_provider_available(cls, provider_key: str) -> bool:
        """Check if a specific provider is available and its dependencies are installed.
        
        Args:
            provider_key: String identifier for the provider
            
        Returns:
            bool: True if the provider is available and can be loaded
            
        Example:
            ```python
            if ProviderFactory.is_provider_available("openai"):
                provider = ProviderFactory.create_provider("openai")
            ```
        """
        try:
            cls.create_provider(provider_key)
            return True
        except (ImportError, ValueError):
            return False 