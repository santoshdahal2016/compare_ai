import pytest
import requests
from unittest.mock import Mock, patch

from actions.run_predictions import RunPrediction
from results_manager import result as Result

def test_run_prediction_execute_success():
    # Mock dependencies
    mock_model = Mock()
    mock_model.get_endpoint.return_value = "http://test-endpoint"
    
    mock_dataset = Mock()
    mock_dataset.get_data.return_value = {"test": "data"}
    
    # Create prediction instance
    prediction = RunPrediction(model=mock_model, dataset=mock_dataset)
    
    # Mock requests.post response
    mock_response = Mock()
    mock_response.json.return_value = {"predictions": [1, 2, 3]}
    
    with patch('requests.post') as mock_post:
        mock_post.return_value = mock_response
        
        # Execute prediction
        result = prediction.execute()
        
        # Verify interactions
        mock_model.get_endpoint.assert_called_once()
        mock_dataset.get_data.assert_called_once()
        mock_post.assert_called_once_with(
            "http://test-endpoint",
            json={"data": {"test": "data"}}
        )
        assert isinstance(result, Result)

def test_run_prediction_execute_api_error():
    # Mock dependencies
    mock_model = Mock()
    mock_model.get_endpoint.return_value = "http://test-endpoint"
    
    mock_dataset = Mock()
    mock_dataset.get_data.return_value = {"test": "data"}
    
    # Create prediction instance
    prediction = RunPrediction(model=mock_model, dataset=mock_dataset)
    
    # Mock requests.post to raise exception
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException("API Error")
        
        # Execute should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            prediction.execute()
        
        assert str(exc_info.value) == "Error calling model endpoint: API Error"

