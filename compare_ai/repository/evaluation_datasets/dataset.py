from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class EvaluationDataset(ABC):
    """Abstract base class representing an evaluation dataset."""
    name: str
    description: str
    entries: List[Dict[str, Any]]
    inputs: Optional[List[str]] = None



    def __post_init__(self):
        """Validate the dataset after initialization."""
        if not self.name:
            raise ValueError("Dataset name cannot be empty")
        if not isinstance(self.entries, list):
            raise ValueError("Entries must be a list")
    
    @abstractmethod
    def load(self) -> None:
        """Abstract method to load the dataset."""
        pass

    @abstractmethod
    def preprocess(self) -> None:
        """Abstract method to transform  dataset to input suitable for model."""
        pass

    @abstractmethod
    def postprocess(self) -> None:
        """Postprocess the output."""
        pass



@dataclass
class TextGenerationEvaluationDataset(EvaluationDataset):
    """Abstract base class for text generation evaluation datasets."""
    # Additional fields specific to text generation datasets
    system_prompts: List[str] = None

    def __post_init__(self):
        """Validate the dataset after initialization."""
        super().__post_init__()
        if self.system_prompts is not None and not isinstance(self.system_prompts, list):
            raise ValueError("system_prompt must be a list of strings")

    @abstractmethod
    def load(self) -> None:
        """Load the dataset."""
        pass

    @abstractmethod
    def preprocess(self) -> List[Dict[str, Any]]:
        """
        Preprocess the dataset to generate inputs suitable for the model.
        Typically includes prompt generation.
        """
        pass

    @abstractmethod
    def postprocess(self, outputs: List[str]) -> List[Dict[str, Any]]:
        """
        Postprocess model outputs.
        Includes cleaning, formatting, and alignment with evaluation metrics.
        """
        pass
