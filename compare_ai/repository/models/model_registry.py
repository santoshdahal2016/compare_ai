from typing import Dict, List, Optional, Set, Any
from .models import Model
from .definitions import TaskType
from .provider_factory import ProviderFactory
from .provider import Provider

class ModelRegistry:
    """Central registry for managing AI model providers and their models.
    
    This class serves as the main interface for discovering and using AI models
    across different providers. It handles provider initialization, model discovery,
    and provides a unified interface for model operations.
    
    Example:
    ```python
    # Initialize with provider configurations
    config = {
        "openai": {"api_key": "your-key"},
        "anthropic": {"api_key": "your-key"}
    }
    registry = ModelRegistry(config)

    # Find models for specific tasks
    text_models = registry.find_models(
        task=TaskType.TEXT_GENERATION,
        modality=Modality.TEXT
    )

    # Use a model
    result = text_models[0].predict("Hello, world!")
    ```
    """

    def __init__(self, config: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize the model registry.
        
        Args:
            config: Optional configuration dictionary for providers
                   Format: {"provider_name": {"config_key": "value"}}
        """
        self._models: Dict[str, Model] = {}
        self._providers: Dict[str, Provider] = {}
        self._load_providers(config or {})

    def _load_providers(self, config: Dict[str, Dict[str, Any]]) -> None:
        """Load available providers using the ProviderFactory.
        
        This method attempts to load all supported providers and their models,
        skipping those that are not installed or cannot be initialized.
        
        Args:
            config: Configuration dictionary for providers
        """
        for provider_name in ProviderFactory.get_supported_providers():
            try:
                provider_config = config.get(provider_name, {})
                provider = ProviderFactory.create_provider(provider_name, provider_config)

                self._providers[provider.name] = provider

                # Auto-register all models from this provider
                self._register_provider_models(provider)
            except ImportError:
                # Skip providers that aren't installed
                continue
            except Exception as e:
                # Log other provider loading errors
                continue

    def _register_provider_models(self, provider: Provider) -> None:
        """Register all available models from a provider.
        
        Args:
            provider: Provider instance to register models from
        """
        for model in provider.get_models():
            self._models[model.model_name] = model

    def get_available_providers(self) -> Set[str]:
        """Get the set of currently loaded providers.
        
        Returns:
            Set[str]: Names of loaded providers
            
        Example:
            ```python
            registry = ModelRegistry()
            providers = registry.get_available_providers()
            print(f"Loaded providers: {providers}")
            ```
        """
        return set(self._providers.keys())



    def find_models(self, 
                   task: TaskType) -> List[Model]:
        """Find models that support the specified task and modality requirements.
        
        Args:
            task: Required TaskType the model must support
            modality: Required Modality the model must support
            
        Returns:
            List[Model]: List of models meeting the requirements
        """
        matching_models = []
        
        # Get relevant providers

        
        # Check each provider's models
        for provider_instance in self._providers.values():
            for model in provider_instance.get_models():                
                if model.supports_task(task):
                    matching_models.append(model)
                    
        return matching_models

    def get_supported_tasks(self) -> Set[TaskType]:
        """Get all supported tasks across available models.
        
        Args:
            modality: Optional modality to filter tasks by
            
        Returns:
            Set[TaskType]: Set of supported tasks
        """
        tasks = set()
        for provider in self._providers.values():
            for model in provider.get_models():
                tasks.update(model.supported_task)
        return tasks