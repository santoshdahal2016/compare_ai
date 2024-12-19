from dataclasses import dataclass
from typing import Dict, Set, Optional, List
from .definitions import Modality, TaskType, ModelMetrics, TaskRequirements

@dataclass
class ModelCapabilities:
    supported_tasks: Set[TaskType]
    supported_modalities: Set[Modality]
    task_requirements: Dict[TaskType, TaskRequirements]
    supported_formats: Dict[Modality, List[str]]
    batch_support: bool
    max_batch_size: Optional[int] = None
    metrics: Optional[Dict[TaskType, ModelMetrics]] = None

class Model:
    def __init__(
        self,
        model_id: str,
        model_name: str,
        provider: str,
        version: str,
        capabilities: ModelCapabilities,
        model_card_url: Optional[str] = None,
    ):
        self._model_id = model_id
        self._model_name = model_name
        self._provider = provider
        self._version = version
        self._capabilities = capabilities
        self._model_card_url = model_card_url

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def version(self) -> str:
        return self._version

    def supports_task(self, task: TaskType) -> bool:
        return task in self._capabilities.supported_tasks

    def get_task_requirements(self, task: TaskType) -> Optional[TaskRequirements]:
        return self._capabilities.task_requirements.get(task)

    def get_task_metrics(self, task: TaskType) -> Optional[ModelMetrics]:
        return self._capabilities.metrics.get(task) if self._capabilities.metrics else None

    def to_dict(self) -> Dict:
        return {
            "model_id": self._model_id,
            "model_name": self._model_name,
            "provider": self._provider,
            "version": self._version,
            "supported_tasks": [task.value for task in self._capabilities.supported_tasks],
            "supported_modalities": [mod.value for mod in self._capabilities.supported_modalities],
            "model_card_url": self._model_card_url
        }
