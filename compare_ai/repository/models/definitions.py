from enum import Enum
from typing import Dict, Set, List, Optional, Any
from dataclasses import dataclass

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
