from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class EvaluationDataset(ABC):
    """Abstract base class representing an evaluation dataset."""
    name: str
    description: str
    entries: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

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
    def validate(self) -> bool:
        """Abstract method to validate the dataset format.
        
        Returns:
            bool: True if the dataset is valid, False otherwise.
        """
        pass


    
    

