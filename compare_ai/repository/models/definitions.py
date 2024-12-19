from enum import Enum, auto
from typing import Dict, Set, List, Optional, Union, Any
from dataclasses import dataclass

class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"

class TaskType(Enum):
    # Text Tasks
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    TEXT_CLASSIFICATION = "text_classification"
    
    # Vision Tasks
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_GENERATION = "image_generation"
    
    # Audio Tasks
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO_CLASSIFICATION = "audio_classification"
    
    # Multimodal Tasks
    VISUAL_QA = "visual_qa"
    IMAGE_CAPTIONING = "image_captioning"

@dataclass
class ModelMetrics:
    accuracy: Optional[float] = None
    latency: Optional[float] = None  # in milliseconds
    throughput: Optional[int] = None  # requests per second
    custom_metrics: Dict[str, Any] = None

@dataclass
class TaskRequirements:
    input_format: Dict[str, Any]
    output_format: Dict[str, Any]
    max_input_size: int
    supported_languages: Optional[List[str]] = None 