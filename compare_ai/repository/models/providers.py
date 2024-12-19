from abc import ABC, abstractmethod
from typing import Optional, Dict
from .models import Model, ModelCapabilities
from .definitions import TaskType, Modality, TaskRequirements, ModelMetrics


class Provider(ABC):
    """Abstract base class for AI model providers.
    
    Defines the interface that all model providers must implement to provide
    consistent access to model information and capabilities.
    
    Example:
        ```python
        # Create a concrete provider implementation
        provider = OpenAIProvider()
        
        # Get model information
        model = provider.get_model("gpt-4")
        
        # Check model capabilities
        capabilities = provider.get_capabilities("gpt-4")
        
        # Get model version
        version = provider.get_version("gpt-4")
        ```
    """
    
    @abstractmethod
    def get_model(self, model_name: str) -> Model:
        """Retrieve model information for the specified model name.
        
        Args:
            model_name (str): Name of the model to retrieve information for
            
        Returns:
            Model: Model instance containing metadata and capabilities
        """
        pass

    @abstractmethod
    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get the capabilities of the specified model.
        
        Args:
            model_name (str): Name of the model to get capabilities for
            
        Returns:
            ModelCapabilities: Object describing the model's capabilities
        """
        pass

    @abstractmethod
    def get_version(self, model_name: str) -> str:
        """Get the version string for the specified model.
        
        Args:
            model_name (str): Name of the model to get version for
            
        Returns:
            str: Version string for the model
        """
        pass

class OpenAIProvider(Provider):
    """Provider implementation for OpenAI models.
    
    This provider handles all OpenAI models including GPT-4, GPT-3.5, and their variants.
    It provides model information, capabilities, and version details specific to OpenAI's offerings.
    
    Example:
        ```python
        # Initialize the OpenAI provider
        openai_provider = OpenAIProvider()
        
        # Get GPT-4 model information
        gpt4_model = openai_provider.get_model("gpt-4")
        
        # Check if the model supports vision tasks
        vision_model = openai_provider.get_model("gpt-4-vision-preview")
        capabilities = vision_model.capabilities
        if TaskType.VISUAL_QA in capabilities.supported_tasks:
            print("Model supports visual Q&A")
        ```
    """
    
    def get_model(self, model_name: str) -> Model:
        """Retrieve OpenAI model information.
        
        Args:
            model_name (str): Name of the OpenAI model (e.g., 'gpt-4')
            
        Returns:
            Model: Model instance with OpenAI-specific metadata
        """
        capabilities = self.get_capabilities(model_name)
        version = self.get_version(model_name)
        model_id = f"openai/{model_name}"
        
        return Model(
            model_id=model_id,
            model_name=model_name,
            provider="openai",
            version=version,
            capabilities=capabilities,
            model_card_url=self._get_model_card_url(model_name)
        )

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for OpenAI models.
        
        Args:
            model_name (str): Name of the OpenAI model
            
        Returns:
            ModelCapabilities: Capabilities object or None if model not found
        """
        if model_name == "gpt-4-vision-preview":
            return ModelCapabilities(
                supported_tasks={
                    TaskType.TEXT_GENERATION,
                    TaskType.VISUAL_QA,
                    TaskType.IMAGE_CAPTIONING
                },
                supported_modalities={
                    Modality.TEXT,
                    Modality.IMAGE
                },
                task_requirements={
                    TaskType.VISUAL_QA: TaskRequirements(
                        input_format={
                            "image": {"max_size": 20971520, "formats": ["png", "jpg", "jpeg"]},
                            "text": {"max_tokens": 4096}
                        },
                        output_format={"text": {"max_tokens": 4096}},
                        max_input_size=20971520
                    )
                },
                supported_formats={
                    Modality.IMAGE: ['png', 'jpg', 'jpeg'],
                    Modality.TEXT: ['txt', 'md', 'json']
                },
                batch_support=True,
                max_batch_size=20,
                metrics={
                    TaskType.VISUAL_QA: ModelMetrics(
                        accuracy=0.95,
                        latency=1000,
                        throughput=10
                    )
                }
            )
        return None

    def get_version(self, model_name: str) -> str:
        """Get version string for OpenAI models.
        
        Args:
            model_name (str): Name of the OpenAI model
            
        Returns:
            str: Version string or 'unknown' if not found
        """
        version_map = {
            "gpt-4": "4.0",
            "gpt-4-vision-preview": "4.0",
            "gpt-3.5-turbo": "3.5",
        }
        return version_map.get(model_name, "unknown")

    def _get_model_card_url(self, model_name: str) -> Optional[str]:
        # Implementation for getting model card URLs
        pass

class AnthropicProvider(Provider):
    """Provider implementation for Anthropic models.
    
    This provider handles all Anthropic models including Claude-3 variants.
    It provides model information, capabilities, and version details specific to Anthropic's offerings.
    
    Example:
        ```python
        # Initialize the Anthropic provider
        anthropic_provider = AnthropicProvider()
        
        # Get Claude-3 model information
        claude_model = anthropic_provider.get_model("claude-3-opus")
        
        # Check supported languages for chat
        capabilities = claude_model.capabilities
        if TaskType.CHAT in capabilities.supported_tasks:
            chat_requirements = capabilities.task_requirements[TaskType.CHAT]
            supported_languages = chat_requirements.supported_languages
            print(f"Supported languages: {supported_languages}")
        ```
    """
    
    def get_model(self, model_name: str) -> Model:
        """Retrieve Anthropic model information.
        
        Args:
            model_name (str): Name of the Anthropic model (e.g., 'claude-3')
            
        Returns:
            Model: Model instance with Anthropic-specific metadata
        """
        capabilities = self.get_capabilities(model_name)
        version = self.get_version(model_name)
        model_id = f"anthropic/{model_name}"
        
        return Model(
            model_id=model_id,
            model_name=model_name,
            provider="anthropic",
            version=version,
            capabilities=capabilities
        )

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for Anthropic models.
        
        Args:
            model_name (str): Name of the Anthropic model
            
        Returns:
            ModelCapabilities: Capabilities object or None if model not found
        """
        if model_name == "claude-3":
            return ModelCapabilities(
                supported_tasks={
                    TaskType.TEXT_GENERATION,
                    TaskType.CHAT,
                    TaskType.VISUAL_QA
                },
                supported_modalities={
                    Modality.TEXT,
                    Modality.IMAGE
                },
                task_requirements={
                    TaskType.CHAT: TaskRequirements(
                        input_format={"text": {"max_tokens": 200000}},
                        output_format={"text": {"max_tokens": 4096}},
                        max_input_size=200000,
                        supported_languages=["en", "es", "fr", "de"]
                    )
                },
                supported_formats={
                    Modality.IMAGE: ['png', 'jpg', 'jpeg', 'gif', 'webp'],
                    Modality.TEXT: ['txt', 'md', 'json']
                },
                batch_support=True,
                max_batch_size=10
            )
        return None

    def get_version(self, model_name: str) -> str:
        """Get version string for Anthropic models.
        
        Args:
            model_name (str): Name of the Anthropic model
            
        Returns:
            str: Version string or 'unknown' if not found
        """
        version_map = {
            "claude-3-opus": "3.0",
            "claude-3-sonnet": "3.0",
            "claude-2.1": "2.1",
        }
        return version_map.get(model_name, "unknown")