import pytest
from unittest.mock import Mock, patch
from compare_ai.repository.models.model_registry import ModelRegistry
from compare_ai.repository.models.definitions import TaskType
from compare_ai.repository.models.models import Model, ModelCapability
from compare_ai.repository.models.provider import Provider

class MockProvider(Provider):
    def __init__(self, provider_name="mock", api_key=None):
        super().__init__(api_key)
        self.provider_name = provider_name
        
    def get_models(self):
        capability = ModelCapability(
            supported_task=TaskType.TEXT_GENERATION,
            supported_formats=["txt"]
        )
        return [
            Model(
                model_name="mock-model",
                provider=self,
                capability=capability
            )
        ]
        
    def _list_available_models(self):
        return ["mock-model"]
        
    def predict(self, model_name, inputs, task):
        return ["mock response"]

class TestModelRegistry:
    @pytest.fixture
    def mock_provider_factory(self):
        with patch('compare_ai.repository.models.model_registry.ProviderFactory') as factory:
            factory.get_supported_providers.return_value = {"mock"}
            factory.create_provider.return_value = MockProvider()
            yield factory
            
    @pytest.fixture
    def registry(self, mock_provider_factory):
        return ModelRegistry()

    def test_initialization(self, registry):
        """Test basic initialization of ModelRegistry"""
        assert isinstance(registry._models, dict)
        assert isinstance(registry._providers, dict)



    def test_find_models_by_task(self, registry):
        """Test finding models by task type"""
        models = registry.find_models(task=TaskType.TEXT_GENERATION)
        assert len(models) > 0
        assert all(model.supports_task(TaskType.TEXT_GENERATION) for model in models)

    def test_find_models_no_match(self, registry):
        """Test finding models with no matches"""
        models = registry.find_models(task=TaskType.IMAGE_GENERATION)
        assert len(models) == 0

    def test_get_supported_tasks(self, registry):
        """Test getting all supported tasks"""
        tasks = registry.get_supported_tasks()
        assert isinstance(tasks, set)
        assert TaskType.TEXT_GENERATION in tasks



    def test_available_providers(self):
        """Test getting available providers"""

        registry = ModelRegistry()
        assert "openai" in registry.get_available_providers()


