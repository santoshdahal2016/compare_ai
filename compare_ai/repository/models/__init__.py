from .definitions import TaskType
from .provider import Provider
from .models import Model, ModelCapability
from .model_registry import ModelRegistry
from .provider_factory import ProviderFactory

__all__ = ['Model', 'ModelCapability', 'Provider', 'TaskType', 'ModelRegistry', 'ProviderFactory']

"""
Models Package
=============

This package provides core components for managing AI models and their configurations.

Key Components:
-------------
- TaskType: Enum defining supported AI tasks (classification, generation, etc.)
- Model: Base class representing an AI model with its metadata and capabilities
- Provider: Class representing an AI model provider/vendor

Example Usage:
------------

To get started, you can create a concrete provider implementation and use it to retrieve model information and capabilities.

```python


registry = ModelRegistry()

text_models = registry.find_models(
    task=TaskType.TEXT_GENERATION)

for model in text_models:
    print(model.predict("Hello, how are you?"))

```
"""

