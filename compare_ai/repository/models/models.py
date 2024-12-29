from dataclasses import dataclass
from typing import Dict, Set, Optional, List, Any, Union
from .definitions import TaskType
from .provider import Provider

@dataclass
class ModelCapability:
    supported_task: TaskType
    supported_formats: List[str]


    def supports_task(self, task: TaskType) -> bool:
        """Check if the model supports both the specified task and modality.

        Args:
            task: TaskType to check

        Returns:
            bool: True if  task is  supported
        """
        return task == self.supported_task 

    def get_supported_formats(self) -> List[str]:
        """Get supported formats for a specific modality.

        Args:
            modality: Modality to get formats for

        Returns:
            List[str]: List of supported format strings
        """
        return self.get("supported_formats", [])


class Model:
    def __init__(
        self,
        model_name: str,
        provider: Provider,
        capability: ModelCapability,
    ):
        if not isinstance(provider, Provider):
            raise TypeError(f"provider must be an instance of Provider, got {type(provider)}")
        
        self._model_name = model_name
        self._provider = provider
        self._capability = capability

  
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> Provider:
        return self._provider

   

    @property
    def supported_task(self) -> TaskType:
        return {self._capability.supported_task}
    

    @property
    def capability(self) -> ModelCapability:
        return self._capability

    def supports_task(self, task: TaskType) -> bool:
        return task == self._capability.supported_task


    def get_supported_formats(self) -> List[str]:
        return self._capability.supported_formats

    def to_dict(self) -> Dict:
        return {
            "model_name": self._model_name,
            "provider": self._provider,
            "capability": {
                "supported_task": self._capability.supported_task.value,
                "supported_formats": self._capability.supported_formats
            }
        }

    def predict(self, 
               input_data: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]) -> Union[Any, List[Any]]:
        """Execute prediction using the model. Always uses batch processing internally.
        
        Args:
            input_data: Single input or list of inputs for prediction
            task: Optional TaskType to specify which task to perform
            
        Returns:
            Union[Any, List[Any]]: Single result or list of results depending on input
            
        Raises:
            ValueError: If inputs are invalid or task is not supported
        """
        task = self._capability.supported_task
            
 
        # Convert single input to list
        is_single_input = not isinstance(input_data, list)
        inputs = [input_data] if is_single_input else input_data
             
        results = self._provider.predict(self.model_name, inputs, task)
        
        # Return single result if single input was provided
        return results[0] if is_single_input else results