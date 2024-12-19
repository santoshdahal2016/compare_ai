from .definitions import Modality, TaskType, ModelMetrics, TaskRequirements
from .models import Model
from .providers import Provider
from .model_registry import ModelRegistry       

__all__ = [
    "Modality", "TaskType", "ModelMetrics", "TaskRequirements",
    "Model", "Provider", "ModelRegistry"
]

"""
Models Package
=============

This package provides core components for managing AI models and their configurations.

Key Components:
-------------
- Modality: Enum defining supported input/output types (text, image, audio, etc.)
- TaskType: Enum defining supported AI tasks (classification, generation, etc.)
- ModelMetrics: Data class for storing model performance metrics
- TaskRequirements: Data class defining requirements for a specific AI task
- Model: Base class representing an AI model with its metadata and capabilities
- Provider: Class representing an AI model provider/vendor
- ModelRegistry: Registry for managing and accessing available models

Example Usage:
------------

To get started, you can create a concrete provider implementation and use it to retrieve model information and capabilities.

```python
# Create a concrete provider implementation
provider = OpenAIProvider()

# Get model information
model = provider.get_model("gpt-4")

# Check model capabilities
capabilities = model.capabilities
```
"""

