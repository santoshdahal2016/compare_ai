from typing import Optional, List, Any, Dict
from ..models import Model, ModelCapability, TaskType
import openai
from ..provider_factory import Provider

class OpenaiProvider(Provider):
    """Provider implementation for OpenAI models.
    
    This provider manages metadata and inference for OpenAI models including GPT-4, GPT-3.5, and their variants.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI provider with API credentials."""
        super().__init__()
        self.client = openai.OpenAI(api_key=api_key)
        self.api_key = api_key
        self.name = "openai"

    def get_models(self) -> List[Model]:
        """Get all models available for this provider.
        

        Returns:
            List of Model instances with their capabilities
        """
        # Get base capabilities for the model

        models = []
        models_array = self._list_available_models()

        for model in models_array:
            for task in model["task_support"]:
                capability = ModelCapability(
                    supported_task=task,
                    supported_formats=self._get_supported_formats(task),
                )
                
            models.append(Model(
                model_name=model["model_name"],
                provider=self,
                capability=capability
            ))
        
        return models

    def _list_available_models(self) -> List[str]:
        """List available OpenAI models."""
        return [
            {
                "model_name": "gpt-4",
                "task_support": [TaskType.TEXT_GENERATION],
            },
            {
                "model_name": "gpt-4-vision-preview",
                "task_support": [TaskType.TEXT_GENERATION]
            },
            {
                "model_name": "gpt-3.5-turbo",
                "task_support": [TaskType.TEXT_GENERATION],
            }
        ]

    def predict(self, model_name: str, inputs: List[Dict[str, Any]], task: TaskType) -> List[Any]:
        """Execute prediction using OpenAI models.
        
        Args:
            model_name: Name of the OpenAI model
            inputs: List of input dictionaries
            task: TaskType to perform
            
        Returns:
            List of model outputs
        """
        if task == TaskType.TEXT_GENERATION:
            return self._predict_text(model_name, inputs)
 

    def _predict_text(self, model_name: str, inputs: List[Dict[str, Any]]) -> List[str]:
        """Handle text generation predictions."""
        responses = []
        for input_data in inputs:
            # Update this section to handle the messages format
            messages = input_data.get("messages", [])
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages  # Use the messages directly instead of creating new message
            )
            responses.append(response.choices[0].message.content)
        return responses


    def _get_supported_formats(self, task: TaskType) -> List[str]:
        """Get supported formats for task."""
        format_map = {
            TaskType.TEXT_GENERATION: ["txt", "md", "json"]
        }
        return format_map.get(task, [])

