import requests
from abc import ABC, abstractmethod
from results_manager import result as Result

class ModelCaller(ABC):
    @abstractmethod
    def call_model(self, endpoint, data):
        pass

class OpenAIModelCaller(ModelCaller):
    def call_model(self, endpoint, data):
        response = requests.post(
            endpoint,
            json={"model": self.model_name, "messages": data},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()

class AnthropicModelCaller(ModelCaller):
    def call_model(self, endpoint, data):
        response = requests.post(
            endpoint,
            json={"model": self.model_name, "prompt": data},
            headers={"x-api-key": self.api_key}
        )
        return response.json()

class RunPrediction:
    MODEL_CALLERS = {
        "openai": OpenAIModelCaller,
        "anthropic": AnthropicModelCaller,
        # Add more model types here
    }

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.model_caller = self._get_model_caller()

    def _get_model_caller(self):
        caller_class = self.MODEL_CALLERS.get(self.model.model_type)
        if not caller_class:
            raise ValueError(f"Unsupported model type: {self.model.model_type}")
        return caller_class()

    def execute(self):
        """
        Executes the prediction using the provided model and dataset.
        """
        test_data = self.dataset.get_data()
        
        try:
            predictions = self.model_caller.call_model(
                self.model.get_endpoint(),
                test_data
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling model endpoint: {e}")

        return Result(predictions)

