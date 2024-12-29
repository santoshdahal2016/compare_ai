from typing import Dict, Any
from datasets import load_dataset
from .dataset import EvaluationDataset

class MMLUDataset(EvaluationDataset):
    """MMLU (Massive Multitask Language Understanding) dataset implementation."""
    
    def __init__(self, subjects=None):
        super().__init__(
            name="MMLU",
            description="Massive Multitask Language Understanding benchmark",
            entries=[],
            metadata={
                "source": "https://huggingface.co/datasets/cais/mmlu",
                "subjects": subjects
            }
        )
        self.subjects = subjects

    def load(self) -> None:
        """Load MMLU dataset from Hugging Face's CAIS format."""
        try:
            # Load the test split from the CAIS MMLU dataset
            dataset = load_dataset("cais/mmlu", split="test")
            
            # Filter subjects if specified
            if self.subjects:
                dataset = dataset.filter(lambda x: x['subject'] in self.subjects)

            # Convert to our format
            self.entries = [{
                'question': item['question'],
                'choices': [item['A'], item['B'], item['C'], item['D']],
                'answer': ['A', 'B', 'C', 'D'][item['answer']],  # Convert numeric answer to letter
                'subject': item['subject']
            } for item in dataset]

            # Update metadata with subject information
            self.metadata.update({
                "total_examples": len(self.entries),
                "subjects_included": list(set(item['subject'] for item in dataset))
            })

        except Exception as e:
            raise RuntimeError(f"Failed to load MMLU dataset: {str(e)}")

    def validate(self) -> bool:
        """Validate MMLU dataset format.
        
        Returns:
            bool: True if the dataset is valid, False otherwise.
        """
        if not self.entries:
            return False

        required_fields = {'question', 'choices', 'answer', 'subject'}
        
        for entry in self.entries:
            if not isinstance(entry, dict):
                return False
            if not all(field in entry for field in required_fields):
                return False
            if not isinstance(entry['choices'], list) or len(entry['choices']) != 4:
                return False
            if not isinstance(entry['answer'], str) or entry['answer'] not in 'ABCD':
                return False
            if not isinstance(entry['subject'], str):
                return False

        return True
