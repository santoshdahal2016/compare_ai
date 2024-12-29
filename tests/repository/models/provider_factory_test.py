import pytest
from unittest.mock import patch, Mock
from compare_ai.repository.models.provider_factory import ProviderFactory
from compare_ai.repository.models.provider import Provider

class TestProviderFactory:
    @pytest.fixture
    def mock_provider_class(self):
        class MockProvider(Provider):
            def __init__(self, api_key=None, **kwargs):
                super().__init__(api_key, **kwargs)
                self.provider_name = "mock"
            
            def get_models(self):
                return []
                
            def _list_available_models(self):
                return []
                
            def predict(self, model_name, inputs, task):
                return []
        
        return MockProvider

    def test_get_supported_providers(self):
        """Test getting supported providers from the providers directory"""
        providers = ProviderFactory.get_supported_providers()
        assert isinstance(providers, set)
        assert "openai" in providers  # We know OpenAI provider exists from the codebase

    def test_create_provider_success(self):
        """Test successful provider creation"""
        config = {"api_key": "test-key"}
        provider = ProviderFactory.create_provider("openai", config)
        assert provider.api_key == "test-key"
        assert provider.name == "openai"

    def test_create_provider_invalid_provider(self):
        """Test error handling for invalid provider"""
        with pytest.raises(ValueError, match="Provider 'invalid' is not supported"):
            ProviderFactory.create_provider("invalid")

    @patch("importlib.import_module")
    def test_create_provider_import_error(self, mock_import):
        """Test handling of import errors when provider package is not installed"""
        mock_import.side_effect = ImportError("Package not found")
        
        with pytest.raises(ImportError, match="Could not import provider 'openai'"):
            ProviderFactory.create_provider("openai")


    def test_is_provider_available(self):
        """Test checking if a provider is available"""
        assert ProviderFactory.is_provider_available("openai") == True
        assert ProviderFactory.is_provider_available("invalid_provider") == False
