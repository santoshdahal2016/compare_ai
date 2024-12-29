import pytest
from unittest.mock import Mock, patch
from compare_ai.repository.models.providers.openai_provider import OpenaiProvider as OpenAIProvider
from compare_ai.repository.models.definitions import TaskType
from dotenv import load_dotenv
import os

load_dotenv()

class TestOpenAIProvider:
    @pytest.fixture
    def provider(self):
        return OpenAIProvider(api_key="test_key")

    @pytest.fixture
    def mock_openai(self):
        with patch("compare_ai.repository.models.providers.openai_provider.openai") as mock:
            yield mock

    def test_initialization(self, provider):
        assert provider.api_key == "test_key"

    def test_get_models(self, provider):
        models = provider.get_models()
        assert len(models) > 0
        
        # Test first model capabilities
        model = models[0]
        assert model.model_name == "gpt-4"
        assert isinstance(model.capability.supported_task, TaskType)
        assert isinstance(model.capability.supported_formats, list)

    def test_list_available_models(self, provider):
        models = provider._list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Check structure of returned model data
        first_model = models[0]
        assert "model_name" in first_model
        assert "task_support" in first_model
        assert isinstance(first_model["task_support"], list)

    def test_predict_text_generation(self, provider, mock_openai):
        # Create a mock client instance
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        
        # Create mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response

        # Set the client explicitly on the provider
        provider.client = mock_client

        inputs = [{"messages": [{"role": "user", "content": "Hello"}]}]
        result = provider.predict("gpt-3.5-turbo", inputs, TaskType.TEXT_GENERATION)
        
        assert result == ["Test response"]
        mock_client.chat.completions.create.assert_called_once()

    def test_predict_visual_qa(self, provider, mock_openai):
        # Create a mock client instance
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        
        # Create mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Image description"))]
        mock_client.chat.completions.create.return_value = mock_response

        # Set the client explicitly on the provider
        provider.client = mock_client

        inputs = [{"messages": [{"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": "image_url"}
        ]}]}]
        result = provider.predict("gpt-4-vision-preview", inputs, TaskType.VISUAL_QA)
        
        assert result == ["Image description"]
        mock_client.chat.completions.create.assert_called_once()

    def test_predict_unsupported_task(self, provider):
        with pytest.raises(ValueError, match="Task .* not supported for model .*"):
            provider.predict("gpt-3.5-turbo", [{"messages": [{"role": "user", "content": "test"}]}], "UNSUPPORTED_TASK")

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key provided")
    def test_batch_text_generation_integration(self):
        inputs = [
            {"messages": [{"role": "user", "content": "What is the capital of France?"}]},
            {"messages": [{"role": "user", "content": "What is the capital of France?"}]},
            {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
        ]
        
        results = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")).predict(
            model_name="gpt-3.5-turbo",
            inputs=inputs,
            task=TaskType.TEXT_GENERATION
        )

        assert len(results) == len(inputs)
        assert all(isinstance(response, str) and response.strip() for response in results)
        assert "Paris" in results[0]
