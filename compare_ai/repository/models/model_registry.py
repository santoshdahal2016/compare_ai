from typing import Dict, List, Optional
from .models import Model

class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, Model] = {}

    def register_model(self, model: Model) -> None:
        self._models[model.model_id] = model

    def get_model(self, model_id: str) -> Optional[Model]:
        return self._models.get(model_id)

    def list_models(self, 
                   provider: Optional[str] = None,
                   task: Optional[TaskType] = None) -> List[Model]:
        models = self._models.values()
        
        if provider:
            models = [m for m in models if m.provider == provider]
            
        if task:
            models = [m for m in models if m.supports_task(task)]
            
        return list(models)