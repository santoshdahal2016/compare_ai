from typing import Dict, Any
from datasets import load_dataset
from .dataset import TextGenerationEvaluationDataset

class MMLUDataset(TextGenerationEvaluationDataset):
    """MMLU (Massive Multitask Language Understanding) dataset implementation."""
    
    def __init__(self,system_prompts=["","You are an expert academic assistant who answers questions precisely and accurately"] ):
        super().__init__(
            name="MMLU",
            description="Massive Multitask Language Understanding benchmark",
            entries=[],
            inputs=[]
        )
        self.system_prompts = system_prompts

        self.load()
        self.preprocess()

    def load(self) -> None:
        """Load MMLU dataset from Hugging Face's CAIS format."""
        try:
            dataset = load_dataset("cais/mmlu", "all")
            self.entries = dataset["test"]
        except Exception as e:
            raise RuntimeError(f"Failed to load MMLU dataset: {str(e)}")
        
    def preprocess(self) -> Dict[str, Any]:
        """Preprocess the MMLU dataset."""


        self.inputs = [
            self._preprocess_entries(system_promt=item) for item in self.system_prompts
        ]        
    
    def _preprocess_entries(self, system_promt="") -> Dict[str, Any]:
        """Preprocess the dataset entries."""


        def generate_prompt(example):
            question = example["question"]
            choices = "\n".join(example["choices"])
            prompt = f"Question: {question}\nOptions:\n{choices}\nAnswer with the correct option (A, B, C, or D)."
            example["prompt"] = prompt
            example["system_prompt"] = system_promt
            return example
        
        return [generate_prompt(entry) for entry in self.entries]

    
    def postprocess(self) -> None:
        """Postprocess the output."""
        pass

