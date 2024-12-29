from abc import ABC, abstractmethod
from typing import List, Optional
from .definitions import TaskType

class Provider(ABC):
    @abstractmethod
    def __init__(self, api_key: str = None, **kwargs):
        self.api_key = api_key
        self.kwargs = kwargs
        self.name: str = ""

    @abstractmethod
    def get_models(self, model_name: Optional[str] = None) -> List[any]:
        pass

    @abstractmethod
    def predict(self, model_name: str, inputs: List[str], task: TaskType) -> List[str]:
        pass